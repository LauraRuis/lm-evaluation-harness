"""
TODO: Add the Paper Title on this line.
TODO: Add the paper's PDF URL (preferably from arXiv) on this line.

TODO: Write a Short Description of the task.

Homepage: TODO: Add the URL to the task's Homepage here.
"""
from lm_eval.api.task import PromptSourceTask
import numpy as np


# TODO: Add the BibTeX citation for the task.
_CITATION = """
"""


class BaseTask(PromptSourceTask):
    # dataset as denoted in HuggingFace `datasets`.
    DATASET_PATH = "UCL-DARK/ludwig"
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = "0-shot"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            return self.dataset["train"]

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def doc_to_target(self, doc: dict):
        _, target = self.prompt_template.apply(doc)
        return [target]


class ZeroShot(BaseTask):
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = "0-shot"


class OneShot(BaseTask):
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = "1-shot"


class FiveShot(BaseTask):
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = "5-shot"


class TenShot(BaseTask):
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = "10-shot"


class FifteenShot(BaseTask):
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = "15-shot"


class ThirtyShot(BaseTask):
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = "30-shot"