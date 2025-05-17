import torch
import torch.nn as nn


class VisualMLP(nn.Module):
    def __init__(self, input_dim=4096):
        super(VisualMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 32)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.dropout1 = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.leaky_relu(x)
        x = self.dropout1(x)
        x = self.fc3(x)

        return x


class AcousticMLP(nn.Module):
    def __init__(self, input_dim=2688):
        super(AcousticMLP, self).__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=1024, bias=True)
        self.fc2 = nn.Linear(in_features=1024, out_features=512, bias=True)
        self.fc3 = nn.Linear(in_features=512, out_features=32, bias=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.dropout1 = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.leaky_relu(x)
        x = self.dropout1(x)
        x = self.fc3(x)

        return x


class TextualMLP(nn.Module):
    def __init__(self, input_dim=1538):
        super(TextualMLP, self).__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=1024, bias=True)
        self.fc2 = nn.Linear(in_features=1024, out_features=512, bias=True)
        self.fc3 = nn.Linear(in_features=512, out_features=32, bias=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.dropout1 = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.leaky_relu(x)
        x = self.dropout1(x)
        x = self.fc3(x)

        return x


class MetadataEncoder(nn.Module):
    def __init__(self, emb_sizes):
        """
        Args:
            emb_sizes: Parameters of the embedding layer. It is a list of tuples,
            where each tuple contains two elements: the number of categories and the embedding dimension
        """
        super(MetadataEncoder, self).__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(num, size) for num, size in emb_sizes])
        self.fc_c = nn.Linear(1, 8)  # Process continuous features
        self.fc = nn.Linear(sum(size for _, size in emb_sizes) + 8, 32)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        embeddings = []
        for i, emb in enumerate(self.embeddings[:21]):
            max_index = x[:, i].max().item()
            num_embeddings = emb.num_embeddings
            if max_index >= num_embeddings:
                raise ValueError(f"MetadataEncoder index out of range for embedding {i}: "
                                 f"max index {max_index} >= {num_embeddings}")
            embeddings.append(emb(x[:, i].long()))
        # Concatenate all the discrete features processed by the embedding layer
        concatenated_embeddings = torch.cat(embeddings, dim=1)
        # Extract continuous features
        continuous_feature = x[:, 21].unsqueeze(1).float()
        processed_continuous_feature = self.leaky_relu(self.fc_c(continuous_feature))

        combined_features = torch.cat([concatenated_embeddings, processed_continuous_feature], dim=1)
        output = self.leaky_relu(self.fc(combined_features))

        return output


class SocialEncoder(nn.Module):
    def __init__(self, emb_sizes):
        super(SocialEncoder, self).__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(num, size) for num, size in emb_sizes])
        embedding_dim = sum(size for _, size in emb_sizes)

        # Process 'user follower count' and 'user following count' to integrate them into 'user social interaction features'
        self.fc1 = nn.Linear(2, 16)
        # Process 'total video views' and 'total video likes' to integrate them into 'user content popularity features'
        self.fc2 = nn.Linear(2, 16)

        dims = embedding_dim + 32
        self.fc = nn.Linear(dims, 32)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        discrete_vars = x[:, :5].long()
        # 'user follower count' and 'user following count' that have been preprocessed by logarithm
        log_follower_following = x[:, 5:7].float()
        # 'total video views' and 'total video likes' that have been preprocessed by logarithm
        log_video_view_like = x[:, 7:9].float()

        embedded_features = []
        for i, emb in enumerate(self.embeddings):
            max_index = discrete_vars[:, i].max().item()
            embedding_size = emb.weight.size(0)
            if max_index >= embedding_size:
                raise ValueError(f"The index (number of categories) {max_index} of SocialEncoder "
                                 f"exceeds the size range of embedding layer {i}: {embedding_size}")
            embedded_features.append(emb(discrete_vars[:, i]))

        embeddings = torch.cat(embedded_features, dim=1)

        fc1_output = self.leaky_relu(self.fc1(log_follower_following))
        fc2_output = self.leaky_relu(self.fc2(log_video_view_like))

        combined_features = torch.cat([embeddings, fc1_output, fc2_output], dim=1)

        output = self.leaky_relu(self.fc(combined_features))

        return output


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.w_p = nn.Linear(hidden_dim, hidden_dim)
        self.b_p = nn.Parameter(torch.zeros(hidden_dim))
        self.q_p = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, x):
        scores = torch.tanh(self.w_p(x) + self.b_p)  # [batch_size, sequence_length, hidden_dim]
        scores = torch.matmul(scores, self.q_p)  # [batch_size, sequence_length]

        # Calculate the attention weights
        attention_weights = torch.softmax(scores, dim=1)  # [batch_size, sequence_length]
        attention_weights = attention_weights.unsqueeze(-1)  # [batch_size, sequence_length, 1]

        f_p = torch.sum(attention_weights * x, dim=1)  # [batch_size, hidden_dim]

        return f_p, attention_weights


class HPTFE(nn.Module):
    def __init__(self,
                 short_term_tcxt_indices, short_term_popularity_tseries_index,
                 long_term_tcxt_indices, long_term_popularity_tseries_index,
                 short_term_tcxt_num_embeddings, short_term_tcxt_embedding_dims,
                 long_term_tcxt_num_embeddings, long_term_tcxt_embedding_dims):
        super(HPTFE, self).__init__()

        self.t_S_indices = short_term_tcxt_indices  # t_S
        self.p_S_index = short_term_popularity_tseries_index  # p_S
        self.t_S_num_embeddings = short_term_tcxt_num_embeddings
        self.t_S_embedding_dims = short_term_tcxt_embedding_dims

        self.t_L_indices = long_term_tcxt_indices  # t_L
        self.p_L_index = long_term_popularity_tseries_index  # p_L
        self.t_L_num_embeddings = long_term_tcxt_num_embeddings
        self.t_L_embedding_dims = long_term_tcxt_embedding_dims

        self.fc_short = nn.Linear(len(self.p_S_index), 8)
        self.fc_long = nn.Linear(len(self.p_L_index), 8)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        # The original dimension of tau_S^k
        self.input_dim_short = 8 + sum(self.t_S_embedding_dims)
        # The original dimension of tau_L^k
        self.input_dim_long = 8 + sum(self.t_L_embedding_dims)

        self.linear_short = nn.Linear(in_features=self.input_dim_short, out_features=64, bias=True)
        self.linear_long = nn.Linear(in_features=self.input_dim_long, out_features=64, bias=True)

        # LSTM_S
        self.lstm_S_1 = nn.LSTM(64, hidden_size=256, num_layers=1, batch_first=True, dropout=0)
        self.lstm_S_2 = nn.LSTM(256, hidden_size=32, num_layers=1, batch_first=True, dropout=0)

        # LSTM_L
        self.lstm_L_1 = nn.LSTM(64 + 32, hidden_size=256, num_layers=1, batch_first=True, dropout=0)
        self.lstm_L_2 = nn.LSTM(256, hidden_size=32, num_layers=1, batch_first=True, dropout=0)

        # Embedding layers
        self.t_S_embeddings = nn.ModuleList([nn.Embedding(num, dim) for num, dim in zip(self.t_S_num_embeddings,
                                                                                        self.t_S_embedding_dims)])
        self.t_L_embeddings = nn.ModuleList([nn.Embedding(num, dim) for num, dim in zip(self.t_L_num_embeddings,
                                                                                        self.t_L_embedding_dims)])

        # Attention pooling
        self.attention_layer = AttentionLayer(32)

    def forward(self, temporal_short, temporal_long):
        """
        Args:
            temporal_short: The original short-term temporal context features and short-term popularity time series
            temporal_long: The original long-term temporal context features and long-term popularity time series
        """
        # Process the short-term popularity trend
        cxt_features_short = temporal_short[:, :, self.t_S_indices].long()
        for i, idx in enumerate(self.t_S_indices):
            max_index = cxt_features_short[:, :, i].max().item()
            if max_index >= self.t_S_num_embeddings[i]:
                raise ValueError(
                    f"Index out of range for embedding {i}: max index {max_index} >= {self.t_S_num_embeddings[i]}")

        t_s = [self.t_S_embeddings[i](cxt_features_short[:, :, i]) for i in range(len(self.t_S_indices))]
        t_s = torch.cat(t_s, dim=-1)

        p_s = temporal_short[:, :, self.p_S_index]
        p_s = self.leaky_relu(self.fc_short(p_s))

        combined_features_short = torch.cat([t_s, p_s], dim=-1)

        tau_s = self.leaky_relu(self.linear_short(combined_features_short))

        lstm_out_1, (h_n_1, c_n_1) = self.lstm_S_1(tau_s)  # lstm_out: [batch_size, sequence_length, hidden_dim]
        lstm_out_1, (h_n_1, c_n_1) = self.lstm_S_2(lstm_out_1)  # lstm_out: [batch_size, sequence_length, hidden_dim]

        # Process the long-term popularity trend
        cxt_features_long = temporal_long[:, :, self.t_L_indices].long()
        for i, idx in enumerate(self.t_L_indices):
            max_index = cxt_features_long[:, :, i].max().item()
            if max_index >= self.t_L_num_embeddings[i]:
                raise ValueError(
                    f"Index out of range for embedding {i}: max index {max_index} >= {self.t_L_num_embeddings[i]}")

        t_l = [self.t_L_embeddings[i](cxt_features_long[:, :, i]) for i in range(len(self.t_L_indices))]
        t_l = torch.cat(t_l, dim=-1)

        p_l = temporal_long[:, :, self.p_L_index]
        p_l = self.leaky_relu(self.fc_long(p_l))
        combined_features_long = torch.cat([t_l, p_l], dim=-1)

        tau_l = self.leaky_relu(self.linear_long(combined_features_long))

        # Extract h_s^12n (expressed as h_sn)
        h_sn = h_n_1[-1].unsqueeze(1).repeat(1, temporal_long.size(1), 1)
        tau_l_h_sn = torch.cat([tau_l, h_sn], dim=-1)

        lstm_out_2, (h_n_2, c_n_2) = self.lstm_L_1(tau_l_h_sn)
        lstm_out_2, (h_n_2, c_n_2) = self.lstm_L_2(lstm_out_2)

        f_p, _ = self.attention_layer(lstm_out_2)

        out = f_p

        return out


class TAFF(nn.Module):
    def __init__(self, attention_hidden_dim):
        super(TAFF, self).__init__()

        self.feature_dim = 32  # The dimension of each feature
        # Define the embedding layer for processing t_P
        self.time_embedding = nn.ModuleList([
            nn.Embedding(2, 2),  # HDE (Holiday Eve, is_day_before_holiday)
            nn.Embedding(2, 2),  # HLD (Holiday, is_holiday)
            nn.Embedding(110, 11)  # PBD (Published Day Count, days_after_upload)
        ])

        self.w_z = nn.Linear(self.feature_dim, attention_hidden_dim, bias=False)
        self.v_z = nn.Linear(15, attention_hidden_dim, bias=False)
        self.q_z = nn.Linear(attention_hidden_dim, 1, bias=False)
        self.b_z = nn.Parameter(torch.zeros(attention_hidden_dim))

    def forward(self, visual_input, acoustic_input, textual_input, metadata_input, social_input,
                temporal_features, t_p):
        # Input t_P into the embedding layer for processing
        t_p = t_p.long()
        t_p_features = [emb(t_p[:, i]) for i, emb in enumerate(self.time_embedding)]
        t_p_features = torch.cat(t_p_features, dim=-1)

        # For each feature (such as visual, acoustic, etc.), calculate its attention score with the temporal context t_P
        features = [visual_input, acoustic_input, textual_input, metadata_input, social_input, temporal_features]

        # Calculate attention weights
        attention_scores = []
        for feature in features:
            a_z = self.q_z(torch.tanh(self.w_z(feature) + self.v_z(t_p_features) + self.b_z))
            attention_scores.append(a_z)

        attention_scores = torch.cat(attention_scores, dim=1)
        attention_weights = torch.softmax(attention_scores, dim=1)

        features = torch.stack(features, dim=1)  # [batch_size, num_features, hidden_dim]
        fused_features = torch.sum(features * attention_weights.unsqueeze(-1), dim=1)

        return fused_features


class PredictionLayer(nn.Module):
    def __init__(self, input_dim):
        super(PredictionLayer, self).__init__()
        self.fc = nn.Linear(in_features=input_dim, out_features=1, bias=True)

    def forward(self, x):
        p_hat = self.fc(x)

        return p_hat


class TIML(nn.Module):
    def __init__(self, attention_hidden_dim, metadata_emb_sizes=None, social_emb_sizes=None,
                 short_term_tcxt_indices=None, short_term_popularity_tseries_index=None,
                 long_term_tcxt_indices=None, long_term_popularity_tseries_index=None,
                 short_term_tcxt_num_embeddings=None, short_term_tcxt_embedding_dims=None,
                 long_term_tcxt_num_embeddings=None, long_term_tcxt_embedding_dims=None):
        super(TIML, self).__init__()

        if metadata_emb_sizes is None:
            metadata_emb_sizes = []
        if social_emb_sizes is None:
            social_emb_sizes = []

        # Initialize all encoders
        self.attention_hidden_dim = attention_hidden_dim
        self.visual_mlp = VisualMLP()
        self.acoustic_mlp = AcousticMLP()
        self.textual_mlp = TextualMLP()
        self.metadata_encoder = MetadataEncoder(metadata_emb_sizes)
        self.social_encoder = SocialEncoder(social_emb_sizes)
        self.temporal_encoder = HPTFE(
            short_term_tcxt_indices=short_term_tcxt_indices,
            short_term_popularity_tseries_index=short_term_popularity_tseries_index,
            long_term_tcxt_indices=long_term_tcxt_indices,
            long_term_popularity_tseries_index=long_term_popularity_tseries_index,
            short_term_tcxt_num_embeddings=short_term_tcxt_num_embeddings,
            short_term_tcxt_embedding_dims=short_term_tcxt_embedding_dims,
            long_term_tcxt_num_embeddings=long_term_tcxt_num_embeddings,
            long_term_tcxt_embedding_dims=long_term_tcxt_embedding_dims
        )

        self.TAFF = TAFF(attention_hidden_dim=self.attention_hidden_dim)

        self.PredictionLayer = PredictionLayer(input_dim=self.attention_hidden_dim)

    def forward(self, visual_input, acoustic_input, textual_input, metadata_input, social_input,
                temporal_short_input, temporal_long_input, t_p):
        visual_features = self.visual_mlp(visual_input)
        acoustic_features = self.acoustic_mlp(acoustic_input)
        textual_features = self.textual_mlp(textual_input)
        metadata_features = self.metadata_encoder(metadata_input)
        social_features = self.social_encoder(social_input)
        temporal_features = self.temporal_encoder(temporal_short_input, temporal_long_input)

        # Apply TAFF
        fused_features = self.TAFF(visual_features, acoustic_features, textual_features, metadata_features,
                                   social_features, temporal_features, t_p)  # [batch_size, feature_dim]

        # Obtain the prediction results through the prediction layer
        output = self.PredictionLayer(fused_features)

        return output
