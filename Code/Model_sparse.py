import pandas as pd
import torch as t
from torch import nn
import torch.nn.functional as F
from Params import args
from Utils.Utils import contrastLoss, ce, l2_norm, pairPredict
from Transformer import Encoder_Layer, TransformerEncoderLayer

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class TransGNN(nn.Module):
    def __init__(self):
        super(TransGNN, self).__init__()

        # Initialize drug and gene embeddings
        self.dEmbeds = nn.Parameter(init(t.empty(args.drug, args.latdim)))
        self.gEmbeds = nn.Parameter(init(t.empty(args.gene, args.latdim)))
        self.drug_transformer_encoder = TransformerEncoderLayer(d_model=args.latdim, num_heads=args.num_head,
                                                                dropout=args.dropout)
        self.gene_transformer_encoder = TransformerEncoderLayer(d_model=args.latdim, num_heads=args.num_head,
                                                                dropout=args.dropout)
        # Initialize classifier layer
        self.classifierLayer = KAN()

        # Initialize SpAdjDropEdge layer for dropout on the graph edges
        self.pickSampEdges = PickSampEdges()

    def drug_transformer_layer(self, embeds, mask=None):
        assert len(embeds.shape) <= 3, "Shape Error, embed shape is {}, out of size!".format(embeds.shape)
        if len(embeds.shape) == 2:
            embeds = embeds.unsqueeze(dim=0)
            embeds = self.drug_transformer_encoder(embeds, mask)
            embeds = embeds.squeeze()
        else:
            embeds = self.drug_transformer_layer(embeds, mask)

        return embeds

    def gene_transformer_layer(self, embeds, mask=None):
        assert len(embeds.shape) <= 3, "Shape Error, embed shape is {}, out of size!".format(embeds.shape)
        if len(embeds.shape) == 2:
            embeds = embeds.unsqueeze(dim=0)
            embeds = self.gene_transformer_encoder(embeds, mask)
            embeds = embeds.squeeze()
        else:
            embeds = self.gene_transformer_layer(embeds, mask)
        return embeds

    def gnn_message_passing(self, adj, embeds):
        if args.data == "DrugBank":
            return l2_norm(t.spmm(adj, embeds))
        else:
            return t.spmm(adj, embeds)
    def forward(self, adj, edgeSampRate):
        # Concatenate drug and gene embeddings
        embeds = t.cat([self.dEmbeds, self.gEmbeds], dim=0)
        embedsLst = [embeds]

        for i in range(args.transgnn_layer):
            # gnn layer
            tmp_embeds = self.gnn_message_passing(self.pickSampEdges(adj, edgeSampRate), embedsLst[-1])

            # Transformer layer
            tmp_drug_embeds = tmp_embeds[:args.drug]
            tmp_gene_embeds = tmp_embeds[args.drug:]
            tmp_drug_embeds = self.drug_transformer_layer(tmp_drug_embeds)
            tmp_gene_embeds = self.gene_transformer_layer(tmp_gene_embeds)
            # add
            tmp_drug_embeds += tmp_embeds[:args.drug]
            tmp_gene_embeds += tmp_embeds[args.drug:]
            tmp_embeds = t.cat([tmp_drug_embeds, tmp_gene_embeds], dim=0)
            embedsLst.append(tmp_embeds)

        # Sum all embeddings
        embeds = sum(embedsLst)
        drug_embeds = embeds[:args.drug]
        gene_embeds = embeds[args.drug:]
        return embeds, drug_embeds, gene_embeds

    def calcLosses(self, drugs, genes, labels, adj, edgeSampRate):
        embeds, drug_embeds, gene_embeds = self.forward(adj, edgeSampRate)
        dEmbeds, gEmbeds = embeds[:args.drug], embeds[args.drug:]

        # Select drug and gene embeddings based on input indices
        dEmbeds = dEmbeds[drugs]
        gEmbeds = gEmbeds[genes]

        # Calculate Cross-Entropy loss
        pre = self.classifierLayer(dEmbeds, gEmbeds)
        ceLoss = ce(pre, labels)
        return ceLoss

    def predict(self, adj, drugs, genes, flag):
        embeds, drug_embeds, gene_embeds = self.forward(adj, 1)
        # dEmbeds, gEmbeds = embeds[:args.drug], embeds[args.drug:]

        allPreds = t.mm(embeds, t.transpose(embeds,1, 0))
        # allPreds = allPreds + 0.3*t.spmm(adj, allPreds)
        allPreds = allPreds + 0.3*adj
        topValues, topLocs = t.topk(allPreds, args.topk, 1)
        topEmbeds = embeds[topLocs]
        weights = topValues.unsqueeze(-1)
        weightedSum = (topEmbeds * weights).sum(dim=1)
        aggregatedEmbeds = weightedSum / weights.sum(dim=1)
        embeds = embeds + args.alpha*aggregatedEmbeds

        dEmbeds, gEmbeds = embeds[:args.drug], embeds[args.drug:]

        # Select drug and gene embeddings based on input indices
        dEmbeds = dEmbeds[drugs]
        gEmbeds = gEmbeds[genes]

        # Perform classification
        pre = self.classifierLayer(dEmbeds, gEmbeds)
        return pre

# Define the SpAdjDropEdge layer for graph edge dropout
class PickSampEdges(nn.Module):
    def __init__(self):
        super(PickSampEdges, self).__init__()

    def forward(self, adj, edgeSampRate):
        if edgeSampRate == 1.0:
            return adj
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        mask = ((t.rand(edgeNum) + edgeSampRate).floor()).type(t.bool)
        newVals = vals[mask] / edgeSampRate
        newIdxs = idxs[:, mask]
        ret = t.sparse.FloatTensor(newIdxs, newVals, adj.shape)
        return ret

class KANClassifierLayer(nn.Module):
    def __init__(self):
        super(KANClassifierLayer, self).__init__()
        self.num_subfunctions = 8
        self.subfunctions = nn.ModuleList([nn.Sequential(
            nn.Linear(args.latdim * 2, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, args.num_classes)
        ) for _ in range(self.num_subfunctions)])

        self.final_layer = nn.Linear(args.num_classes * self.num_subfunctions, args.num_classes)

    def forward(self, dEmbeds, gEmbeds):
        embeds = t.cat((dEmbeds, gEmbeds), dim=1)
        subfunction_outputs = [subfunc(embeds) for subfunc in self.subfunctions]
        subfunction_outputs = t.cat(subfunction_outputs, dim=1)

        output = self.final_layer(subfunction_outputs)
        return output