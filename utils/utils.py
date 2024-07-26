import subprocess

def get_last_commit_hash():
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE, text=True)
    commit_hash = result.stdout.strip()
    return commit_hash