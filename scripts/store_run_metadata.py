'''A script to store metadata about the current run, including git commit hash, timestamp, and other relevant information.
This can help with reproducibility and tracking the state of the codebase when the run was executed.'''
import git
import json
from datetime import datetime

def save_run_metadata(output_file):
    repo = git.Repo(search_parent_directories=True)

    metadata = {
        'timestamp': datetime.now().isoformat(),
        'commit_hash': repo.head.object.hexsha,
        'branch': repo.active_branch.name,
        'git_dirty': repo.is_dirty(), # True if there are uncommitted changes
        'script': __file__
    }

    meta_file= output_file.replace('.csv', '_metadata.json')
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=4)
        print(f"Run metadata saved to {meta_file}")
    return metadata