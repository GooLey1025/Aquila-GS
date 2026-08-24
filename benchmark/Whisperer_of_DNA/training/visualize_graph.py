import torch
import argparse
import yaml
import json
from pathlib import Path
import logging
import copy
from torchviz import make_dot
import warnings
from functools import partial # 导入 partial

# 导入自定义模块
from models.model import DNAWhisper
from utils.config_utils import load_config # 假设你有这个工具函数

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 全局字典来存储梯度信息
gradient_info = {}
nan_gradients = []

def modify_config_recursive(config_dict, key_to_modify, new_value):
    """递归地修改配置字典中特定键的值"""
    if isinstance(config_dict, dict):
        for key, value in config_dict.items():
            if key == key_to_modify and isinstance(value, dict):
                # 修改 gradient_checkpointing 下的 enabled
                if 'enabled' in value:
                    logging.info(f"找到并禁用梯度检查点: {key}")
                    value['enabled'] = new_value
                # 递归查找其他 gradient_checkpointing 设置 (以防万一)
                modify_config_recursive(value, key_to_modify, new_value)
            elif isinstance(value, dict):
                modify_config_recursive(value, key_to_modify, new_value)
            elif isinstance(value, list):
                for item in value:
                    modify_config_recursive(item, key_to_modify, new_value)
    elif isinstance(config_dict, list):
        for item in config_dict:
            modify_config_recursive(item, key_to_modify, new_value)

def module_gradient_hook(module, grad_input, grad_output, name):
    """用于记录模块输出梯度的钩子函数"""
    has_nan = False
    norms = []
    output_idx = 0
    if isinstance(grad_output, tuple):
        for i, grad in enumerate(grad_output):
            if grad is not None and isinstance(grad, torch.Tensor):
                try:
                    norm = grad.norm().item()
                    norms.append(norm)
                    if torch.isnan(grad).any():
                        has_nan = True
                        grad_key = f"{name}_out{i}"
                        if grad_key not in nan_gradients:
                             nan_gradients.append(grad_key)
                        logging.warning(f"检测到 NaN 梯度 (模块输出): {grad_key}")
                except Exception as e:
                    logging.error(f"计算梯度范数时出错 (模块 {name}, 输出 {i}): {e}")
                    norms.append(float('nan')) # 记录错误
            else:
                norms.append(None) # 记录 None 梯度
            output_idx = i
    elif grad_output is not None and isinstance(grad_output, torch.Tensor):
         try:
             norm = grad_output.norm().item()
             norms.append(norm)
             if torch.isnan(grad_output).any():
                 has_nan = True
                 grad_key = f"{name}_out0"
                 if grad_key not in nan_gradients:
                     nan_gradients.append(grad_key)
                 logging.warning(f"检测到 NaN 梯度 (模块输出): {grad_key}")
         except Exception as e:
             logging.error(f"计算梯度范数时出错 (模块 {name}, 输出 0): {e}")
             norms.append(float('nan'))

    first_norm = next((n for n in norms if n is not None), None)
    gradient_info[name] = {'norm': first_norm, 'has_nan': has_nan, 'all_norms': norms}


def visualize(args):
    """可视化模型计算图并检查梯度"""

    # --- 1. 加载和修改配置 ---
    logging.info(f"加载模型配置: {args.model_config}")
    model_config_orig = load_config(args.model_config)
    if not model_config_orig:
        logging.error("无法加载模型配置。")
        return

    # 创建配置副本以进行修改
    model_config = copy.deepcopy(model_config_orig)

    logging.info("禁用所有梯度检查点...")
    modify_config_recursive(model_config, "gradient_checkpointing", False)

    # 确保日志记录配置存在并启用梯度范数记录（虽然我们用hook，但保持一致性）
    if "logging" not in model_config:
        model_config["logging"] = {}
    model_config["logging"]["log_gradient_norms"] = True
    model_config["logging"]["gradient_norm_log_interval"] = 1 # 每次都记录（如果模型内部使用）
    logging.info("在模型配置中设置 log_gradient_norms=True")

    # 提供虚拟的优化器和调度器配置 (模型初始化需要)
    dummy_optimizer_config = {"type": "adamw", "learning_rate": 1e-4}
    dummy_scheduler_config = {"type": "cosine"}

    # --- 2. 实例化模型 ---
    logging.info("实例化 DNAWhisper 模型...")
    try:
        with warnings.catch_warnings():
             warnings.simplefilter("ignore")
             model = DNAWhisper(
                 config=model_config,
                 optimizer_config=dummy_optimizer_config,
                 scheduler_config=dummy_scheduler_config
             )
        model.eval()
        logging.info("模型实例化成功。")
    except Exception as e:
        logging.error(f"模型实例化失败: {e}", exc_info=True)
        return

    # --- 3. 记录可训练参数 ---
    logging.info("记录可训练参数:")
    total_params = 0
    trainable_count = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            logging.info(f"  - {name} (大小: {param.shape}, 数量: {param.numel()})")
            trainable_count += param.numel()
        else:
            logging.info(f"  - {name} (不可训练)")
    logging.info(f"总参数数量: {total_params}")
    logging.info(f"可训练参数数量: {trainable_count}")

    # --- 4. 为模块注册梯度钩子 ---
    logging.info("为模型模块注册后向梯度钩子...")
    hook_handles = []
    for name, module in model.named_modules():
        handle = module.register_full_backward_hook(partial(module_gradient_hook, name=name))
        hook_handles.append(handle)
    logging.info(f"已为 {len(hook_handles)} 个模块注册梯度钩子。")


    # --- 5. 创建虚拟输入和目标 ---
    batch_size = 16
    seq_len = 64
    phenotype_dim = model.hparams.config.get("output_layer", {}).get("phenotype_dim", 3)

    expected_embedding_input = 15
    logging.info(f"使用已知的 Embedding 输入维度: {expected_embedding_input}")
    input_dim = expected_embedding_input

    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    dummy_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    dummy_target = torch.randn(batch_size, phenotype_dim)

    logging.info(f"创建虚拟输入: shape={dummy_input.shape}")
    logging.info(f"创建虚拟掩码: shape={dummy_mask.shape}")
    logging.info(f"创建虚拟目标: shape={dummy_target.shape}")

    # --- 6. 前向传播和计算损失 ---
    logging.info("执行前向传播...")
    loss = None # 初始化 loss 变量
    try:
        outputs = model(dummy_input, mask=dummy_mask)
        phenotype_pred = outputs.get("phenotype_pred")

        if phenotype_pred is None:
            logging.error("模型输出中未找到 'phenotype_pred'。")
            for handle in hook_handles:
                handle.remove()
            return

        loss = torch.nn.functional.mse_loss(phenotype_pred, dummy_target)
        logging.info(f"前向传播完成。计算得到的损失: {loss.item()}")

    except Exception as e:
        logging.error(f"前向传播或损失计算失败: {e}", exc_info=True)
        for handle in hook_handles:
            handle.remove()
        return

    # --- 7. 生成计算图 (移到 backward 之前) ---
    if loss is not None: # 确保 loss 已成功计算
        logging.info("生成计算图 (可能需要一些时间)...")
        output_graph_path = Path(args.output_dir) / f"{Path(args.model_config).stem}_computation_graph"
        try:
            graph = make_dot(loss, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
            graph.render(str(output_graph_path), format="pdf", view=False, cleanup=True)
            logging.info(f"计算图已保存为 PDF: {output_graph_path}.pdf")
            graph.render(str(output_graph_path), format="png", view=False, cleanup=True)
            logging.info(f"计算图已保存为 PNG: {output_graph_path}.png")
        except ImportError:
             logging.error("无法导入 'graphviz'。请安装 graphviz Python 包和系统库 (例如，sudo apt-get install graphviz)。无法生成计算图。")
        except Exception as e:
            logging.error(f"生成计算图失败: {e}", exc_info=True) # 错误信息现在会在这里报告
    else:
        logging.error("无法生成计算图，因为损失计算失败。")


    # --- 8. 反向传播 (现在在生成图之后) ---
    logging.info("执行反向传播以计算梯度...")
    gradient_info.clear()
    nan_gradients.clear()
    try:
        if loss is not None: # 只有在 loss 存在时才进行反向传播
            loss.backward() # 不需要 retain_graph=True 了
            logging.info("反向传播完成。")
        else:
            logging.warning("跳过反向传播，因为损失计算失败。")
    except Exception as e:
        logging.error(f"反向传播失败: {e}", exc_info=True)
        pass # 即使失败，也继续执行 finally 块
    finally:
        logging.info("移除模块梯度钩子...")
        for handle in hook_handles:
            handle.remove()
        logging.info("模块梯度钩子已移除。")


    # --- 9. 报告梯度情况 (现在是第 9 步) ---
    logging.info("--- 梯度报告 (模块输出梯度) ---")
    if not gradient_info:
        logging.warning("未能收集到任何模块梯度信息。可能是反向传播未执行或失败。")
    else:
        logging.info(f"共检查 {len(gradient_info)} 个模块的输出梯度。")
        num_nan = len(nan_gradients)
        if num_nan > 0:
            logging.warning(f"检测到 {num_nan} 个模块输出存在 NaN 梯度:")
            # 提取并排序出现 NaN 的唯一模块名
            unique_module_names_with_nan = sorted(list(set(name.split('_out')[0] for name in nan_gradients)))
            for module_name in unique_module_names_with_nan:
                 logging.warning(f"  - {module_name}")
        else:
            logging.info("未检测到 NaN 模块输出梯度。")

        # 打印所有模块的第一个非 None 输出梯度范数
        logging.info("--- 模块输出梯度范数 (首个非 None) ---") # 添加标题
        for name, info in sorted(gradient_info.items()): # 按模块名称排序
            norm_value = info.get('norm') # 获取范数值

            # 首先检查 norm_value 是否为 None
            if norm_value is None:
                logging.info(f"  模块 {name}: None (无梯度输出)")
            # 然后检查是否为 float 类型的 NaN (计算错误时可能存储为 float('nan'))
            elif isinstance(norm_value, float) and torch.isnan(torch.tensor(norm_value)):
                 logging.info(f"  模块 {name}: 计算错误 (NaN)")
            # 如果不是 None 也不是 NaN，则格式化并打印
            else:
                 try:
                     # 确保可以格式化为浮点数
                     logging.info(f"  模块 {name}: {float(norm_value):.4f}")
                 except ValueError:
                     logging.error(f"  模块 {name}: 无法格式化范数值 {norm_value}")
                     logging.info(f"  模块 {name}: 原始值 {norm_value}")


    logging.info("--- 梯度报告结束 ---")


def main():
    parser = argparse.ArgumentParser(description="可视化 DNAWhisper 模型计算图并检查梯度")
    parser.add_argument("--model-config", type=str, default="training/config/model_config.json",
                        help="模型配置文件路径")
    parser.add_argument("--output-dir", type=str, default="visualization_output",
                        help="保存计算图和其他输出的目录")

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    visualize(args)

if __name__ == "__main__":
    main()