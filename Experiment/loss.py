import torch
from torch.nn import functional as F


def similarity(features_1, features_2, batch, device, temperature):
        
        # labels = torch.cat([torch.arange(batch, dtype=torch.float64)], dim=0).to(device)
        
        # labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        # labels = labels.to(device)
        
        # features_1 = F.normalize(features_1, dim=1)
        # features_2 = F.normalize(features_2, dim=1)

        features_1 /= torch.norm(features_1, dim=1, p=2).unsqueeze(dim=1).to(device)
        features_2 /= torch.norm(features_2, dim=1, p=2).unsqueeze(dim=1).to(device)
        similarity_matrix = torch.matmul(features_1, features_2.T)


        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.batch_size, self.args.n_views * self.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix

        # mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)

        # labels = labels[~mask].view(labels.shape[0], -1)
        # similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # # assert similarity_matrix.shape == labels.shape

        # # select and combine multiple positives
        # positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # # select only the negatives the negatives
        # negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        # logits = torch.cat([positives, negatives], dim=1)
        # labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
        # logits = torch.exp(similarity_matrix)
        # logits = logits / temperature
        return similarity_matrix


