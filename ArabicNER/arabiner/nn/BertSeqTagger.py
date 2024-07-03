import torch.nn as nn
from transformers import BertModel
import torch


class BertSeqTagger(nn.Module):
    def __init__(self, bert_model, num_labels=51, conv_output_channels=64, dropout=0.1, lstm_hidden_size=128):
        super(BertSeqTagger, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(dropout)

        # Convolutional layers setup
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=768, out_channels=conv_output_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=768, out_channels=conv_output_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.Dropout(dropout)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=768, out_channels=conv_output_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.Dropout(dropout)
        )

        self.bilstm = nn.GRU(input_size=768,
                               hidden_size=lstm_hidden_size,
                               num_layers=1,
                               batch_first=True,
                               bidirectional=True)

        # Multihead Attention mechanism for BiLSTM output
        self.attention1 = nn.MultiheadAttention(embed_dim=lstm_hidden_size, num_heads=2, batch_first=True)
        self.attention2 = nn.MultiheadAttention(embed_dim=conv_output_channels, num_heads=2, batch_first=True)
        self.linear1 = nn.Linear(conv_output_channels*3, conv_output_channels)
        self.linear2 = nn.Linear(2 * lstm_hidden_size, lstm_hidden_size)
        self.layer_norm1 = nn.LayerNorm(128)
        self.layer_norm2 = nn.LayerNorm(64)
        self.fc = nn.Linear(conv_output_channels + lstm_hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        # seq_length = sequence_output.size(1)

        lstm_out, _ = self.bilstm(sequence_output)
        lstm_out = self.dropout(lstm_out)
        lstm_out = self.linear2(lstm_out)
        attn_out1, _ = self.attention1(lstm_out, lstm_out, lstm_out)
        # attn_out1 = lstm_out
        attn_out1 = self.layer_norm1(attn_out1)
        sequence_output_permuted = sequence_output.permute(0, 2, 1)
        c1 = self.conv_block1(sequence_output_permuted)
        c2 = self.conv_block2(sequence_output_permuted)
        c3 = self.conv_block3(sequence_output_permuted)
        concatenated = torch.cat((c1, c2, c3), 1)
        concatenated_permuted = self.dropout(concatenated)
        concatenated_permuted = concatenated.permute(0, 2, 1)
        concatenated_permuted = self.linear1(concatenated_permuted)
        

        # print(f"{concatenated_permuted.shape = }")
        attn_out2, _ = self.attention2(concatenated_permuted, concatenated_permuted, concatenated_permuted)
        # print(f"{attn_out2.shape = }",f"{attn_out1.shape = }")
        # attn_out2 = concatenated_permuted
        attn_out2 = self.layer_norm2(attn_out2)
        combined = torch.cat((attn_out1, attn_out2), dim=2)
        # combined = attn_out2
        combined_dropout = self.dropout(combined)
        # print(f"{combined_dropout.shape = }")
        logits = self.fc(combined_dropout)

        return logits
