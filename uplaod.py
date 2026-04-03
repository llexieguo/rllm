from huggingface_hub import login, create_repo, upload_folder
import os

# ===== 改这里 =====
HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = "lexie1218/acc_"
LOCAL_DIR = "/mnt/gpfs/xuexiangyuan/workspace/rllm/checkpoints/mas_orchestra_offline/qwen3_vl_8b_offline_reinforce_cost_reward/global_step_539_model"
PRIVATE = True
# ==================

# 登录
login(token=HF_TOKEN)

# 创建仓库（已存在也不报错）
create_repo(repo_id=REPO_ID, repo_type="model", private=PRIVATE, exist_ok=True)

# 上传整个目录
upload_folder(
    repo_id=REPO_ID,
    folder_path=LOCAL_DIR,
    repo_type="model",
)

print(f"Uploaded to: https://huggingface.co/{REPO_ID}")