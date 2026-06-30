import logging
from pathlib import Path
import git

logger = logging.getLogger(__name__)


class RepoFetcher:
    def __init__(self, clone_dir: str):
        self.clone_dir = Path(clone_dir)

    def fetch(self, repo_url: str) -> Path:
        if self.clone_dir.exists() and any(self.clone_dir.iterdir()):
            logger.info("Repo exists at %s — pulling latest", self.clone_dir)
            try:
                git.Repo(self.clone_dir).remotes.origin.pull()
            except Exception as e:
                logger.warning("Pull failed (%s), using existing clone", e)
        else:
            logger.info("Cloning %s", repo_url)
            self.clone_dir.mkdir(parents=True, exist_ok=True)
            git.Repo.clone_from(repo_url, self.clone_dir, depth=1)
            logger.info("Clone complete")
        return self.clone_dir
