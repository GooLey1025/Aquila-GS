import os
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Any, Optional

def plot_training_curves(log_path: str, 
                         metrics: List[str] = ['train_loss', 'val_loss'],
                         title: str = 'Training and Validation Loss',
                         output_path: Optional[str] = None):
    """
    绘制训练过程中的指标曲线
    
    Args:
        log_path: TensorBoard日志文件路径
        metrics: 需要绘制的指标列表
        title: 图表标题
        output_path: 输出图片路径，如果为None则显示图片
    """
    try:
        # 尝试从TensorBoard日志文件中读取数据
        from tensorboard.backend.event_processing import event_accumulator
        ea = event_accumulator.EventAccumulator(log_path)
        ea.Reload()
        
        plt.figure(figsize=(10, 6))
        
        for metric in metrics:
            if metric in ea.scalars.Keys():
                events = ea.Scalars(metric)
                steps = [event.step for event in events]
                values = [event.value for event in events]
                plt.plot(steps, values, label=metric)
                
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        
        if output_path:
            plt.savefig(output_path)
            print(f"曲线图已保存到 {output_path}")
        else:
            plt.show()
            
    except Exception as e:
        print(f"绘制训练曲线时出错: {e}")
        
        # 尝试直接从CSV文件读取（如果存在）
        csv_path = os.path.join(log_path, 'metrics.csv')
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                
                plt.figure(figsize=(10, 6))
                
                for metric in metrics:
                    if metric in df.columns:
                        plt.plot(df['step'], df[metric], label=metric)
                        
                plt.xlabel('Steps')
                plt.ylabel('Loss')
                plt.title(title)
                plt.legend()
                plt.grid(True)
                
                if output_path:
                    plt.savefig(output_path)
                    print(f"曲线图已保存到 {output_path}")
                else:
                    plt.show()
                    
            except Exception as e2:
                print(f"尝试从CSV读取数据时出错: {e2}")