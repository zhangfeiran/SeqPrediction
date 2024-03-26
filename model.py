
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as tf

try:
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    torch.cuda.get_device_properties(device)
except:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Encoder(nn.Module):
    def __init__(self, input_size=22, hidden_size=64, num_layers=1, dropout=0, T=12):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.T = T
        self.lstm_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)

    def forward(self, input_data):
        input_encoded = torch.zeros(input_data.size(0), self.T, self.hidden_size).to(device)
        hidden = torch.zeros(self.num_layers, input_data.size(0), self.hidden_size).to(device)
        cell = torch.zeros(self.num_layers, input_data.size(0), self.hidden_size).to(device)
        for t in range(self.T):
            weighted_input = input_data[:, t, :]
            self.lstm_layer.flatten_parameters()
            output, (hidden, cell) = self.lstm_layer(weighted_input.unsqueeze(0), (hidden, cell))
            input_encoded[:, t, :] = hidden[self.num_layers - 1]
        return input_encoded


class Decoder(nn.Module):
    def __init__(self, encoder_hidden_size=64, decoder_hidden_size=64, num_layers=1, T=12, dropout=0, out_feats=11):
        super(Decoder, self).__init__()
        self.T = T
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.num_layers = num_layers
        self.lstm_layer = nn.LSTM(input_size=out_feats, hidden_size=decoder_hidden_size, num_layers=num_layers, dropout=dropout)
        self.fc = nn.Linear(encoder_hidden_size + out_feats, out_feats)
        self.fc_final = nn.Linear(decoder_hidden_size + encoder_hidden_size, out_feats)
        self.fc.weight.data.normal_()

    def forward(self, input_encoded, y_hists, y_targs=None, teacher_force=0.5):
        hidden = torch.zeros(self.num_layers, input_encoded.size(0), self.decoder_hidden_size).to(device)
        cell = torch.zeros(self.num_layers, input_encoded.size(0), self.decoder_hidden_size).to(device)
        context = torch.zeros(input_encoded.size(0), self.encoder_hidden_size).to(device)
        for t in range(self.T):
            context = input_encoded.mean(1)
            y_tilde = self.fc(torch.cat((context, y_hists[:, t, :]), dim=1))
            self.lstm_layer.flatten_parameters()
            _, (hidden, cell) = self.lstm_layer(y_tilde.unsqueeze(0), (hidden, cell))
        y_preds = torch.zeros(input_encoded.size(0), 6, 11).to(device)
        y_pred = self.fc_final(torch.cat((hidden[self.num_layers - 1], context), dim=1))
        y_preds[:, 0, :] = y_pred

        for t in range(1, 6):
            context = input_encoded.mean(1)
            y = y_targs[:, t - 1, :] if torch.rand(1) < teacher_force else y_preds[:, t - 1, :]
            y_tilde = self.fc(torch.cat((context, y), dim=1))
            self.lstm_layer.flatten_parameters()
            _, (hidden, cell) = self.lstm_layer(y_tilde.unsqueeze(0), (hidden, cell))
            y_pred = self.fc_final(torch.cat((hidden[self.num_layers - 1], context), dim=1))
            y_preds[:, t, :] = y_pred
        return y_preds



class AttnEncoder(nn.Module):
    def __init__(self, input_size=22, hidden_size=64, num_layers=1, dropout=0, T=12):
        super(AttnEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.T = T
        self.lstm_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.attn_linear = nn.Linear(in_features=2 * hidden_size + T, out_features=1)

    def forward(self, input_data):
        input_encoded = torch.zeros(input_data.size(0), self.T, self.hidden_size).to(device)
        hidden = torch.zeros(self.num_layers, input_data.size(0), self.hidden_size).to(device)
        cell = torch.zeros(self.num_layers, input_data.size(0), self.hidden_size).to(device)
        for t in range(self.T):
            x = torch.cat(
                (
                    hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                    cell.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                    input_data.permute(0, 2, 1),
                ),
                dim=2,
            )
            x = self.attn_linear(x.view(-1, self.hidden_size * 2 + self.T))
            attn_weights = tf.softmax(x.view(-1, self.input_size), dim=1)
            weighted_input = torch.mul(attn_weights, input_data[:, t, :])
            self.lstm_layer.flatten_parameters()
            output, (hidden, cell) = self.lstm_layer(weighted_input.unsqueeze(0), (hidden, cell))
            input_encoded[:, t, :] = hidden[self.num_layers - 1]
        return input_encoded



class AttnDecoder(nn.Module):
    def __init__(self, encoder_hidden_size=64, decoder_hidden_size=64, num_layers=1, T=12, dropout=0, out_feats=11):
        super(AttnDecoder, self).__init__()
        self.T = T
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.num_layers = num_layers
        self.attn_layer = nn.Sequential(
            nn.Linear(2 * decoder_hidden_size + encoder_hidden_size, encoder_hidden_size), nn.Tanh(), nn.Linear(encoder_hidden_size, 1),
        )
        self.lstm_layer = nn.LSTM(input_size=out_feats, hidden_size=decoder_hidden_size, num_layers=num_layers, dropout=dropout)
        self.fc = nn.Linear(encoder_hidden_size + out_feats, out_feats)
        self.fc_final = nn.Linear(decoder_hidden_size + encoder_hidden_size, out_feats)
        self.fc.weight.data.normal_()
    def init_hidden(self, x):
        return torch.zeros(self.num_layers, x.size(0), self.decoder_hidden_size).to(device)
    def forward(self, input_encoded, y_hists, y_targs=None, teacher_force=0.5):
        hidden = self.init_hidden(input_encoded)
        cell = self.init_hidden(input_encoded)
        context = torch.zeros(input_encoded.size(0), self.encoder_hidden_size).to(device)
        for t in range(self.T):
            x = torch.cat(
                (hidden.repeat(self.T, 1, 1).permute(1, 0, 2), cell.repeat(self.T, 1, 1).permute(1, 0, 2), input_encoded,),
                dim=2,
            )
            x = tf.softmax(
                self.attn_layer(x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size)).view(-1, self.T), dim=1,
            )
            context = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :]
            y_tilde = self.fc(torch.cat((context, y_hists[:, t, :]), dim=1))
            self.lstm_layer.flatten_parameters()
            _, (hidden, cell) = self.lstm_layer(y_tilde.unsqueeze(0), (hidden, cell))
        y_preds = torch.zeros(input_encoded.size(0), 6, 11).to(device)
        y_pred = self.fc_final(torch.cat((hidden[self.num_layers - 1], context), dim=1))
        y_preds[:, 0, :] = y_pred
        for t in range(1, 6):
            x = torch.cat((hidden.repeat(self.T, 1, 1).permute(1, 0, 2), cell.repeat(self.T, 1, 1).permute(1, 0, 2), input_encoded,), dim=2,)
            x = tf.softmax(
                self.attn_layer(x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size)).view(-1, self.T), dim=1,
            )
            context = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :]
            y = y_targs[:, t - 1, :] if torch.rand(1) < teacher_force else y_preds[:, t - 1, :]
            y_tilde = self.fc(torch.cat((context, y), dim=1))
            self.lstm_layer.flatten_parameters()
            _, (hidden, cell) = self.lstm_layer(y_tilde.unsqueeze(0), (hidden, cell))
            y_pred = self.fc_final(torch.cat((hidden[self.num_layers - 1], context), dim=1))
            y_preds[:, t, :] = y_pred
        return y_preds
