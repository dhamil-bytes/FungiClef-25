from torch.utils.data import Sampler
import random
from collections import defaultdict, Counter

class EpisodicBatchSampler(Sampler):
    def __init__(self, dataset, num_episodes, num_classes_per_episode, num_support, num_query):
        """
        dataset: the dataset object (which contains the labels)
        num_episodes: number of episodes per epoch
        num_classes_per_episode: N-way
        num_support: K-shot
        num_query: number of query examples per class
        """
        self.dataset = dataset
        self.num_episodes = num_episodes
        self.num_classes_per_episode = num_classes_per_episode
        self.num_support = num_support
        self.num_query = num_query


        # Count occurrences
        label_counts = Counter(self.dataset.labels)
        # Get valid classes with enough examples
        valid_classes = [label for label, count in label_counts.items() if count >= (num_support + num_query)]
        # Filter your dataset accordingly

        # Group data by class label (i.e., all indices for each label)
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(dataset.labels):
            if label in valid_classes:
                self.class_to_indices[label].append(idx)

        # List of all unique class labels in the dataset
        self.classes = list(self.class_to_indices.keys())

    def __len__(self):
        return self.num_episodes

    def __iter__(self):
        for _ in range(self.num_episodes):
            # Sample N classes (e.g., 5-way)
            episode_classes = random.sample(self.classes, self.num_classes_per_episode)
            episode_indices = []

            for cls in episode_classes:
                # Get all examples of the class
                indices = self.class_to_indices[cls]

                # Sample K support + Q query examples
                chosen = random.sample(indices, self.num_support + self.num_query)

                episode_indices.extend(chosen)

            yield episode_indices
