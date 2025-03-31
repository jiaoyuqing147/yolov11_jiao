
class BaseTrainer:
    def __init__(self):
        print("👷 BaseTrainer 初始化中...")

class DetectionTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()  # 必须先初始化父类
        print("🚀 DetectionTrainer 初始化完成")



trainer = DetectionTrainer()
