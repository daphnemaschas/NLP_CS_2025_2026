import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 1. Blocs d'aide (Attention, FFN, Positional Encoding) ---

class MultiHeadAttention(nn.Module):
    """Implémente le mécanisme Multi-Head Self-Attention"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads # Dimension de la tête (d_k)
        self.num_heads = num_heads
        self.d_model = d_model

        # Couches linéaires pour Q, K, V et la sortie
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. Couches linéaires et redimensionnement pour les têtes
        q = self.q_linear(q).view(q.size(0), -1, self.num_heads, self.d_k).transpose(1, 2) # Shape: Batch Size, Seq Length, d_model et Avant le transpose: Batch Size, Seq Length, num_heads, d_model
        k = self.k_linear(k).view(k.size(0), -1, self.num_heads, self.d_k).transpose(1, 2) # Shape: Batch Size, Seq Length, d_model et Avant le transpose: Batch Size, Seq Length, num_heads, d_model
        v = self.v_linear(v).view(v.size(0), -1, self.num_heads, self.d_k).transpose(1, 2) # Shape: Batch Size, Seq Length, d_model et Avant le transpose: Batch Size, Seq Length, num_heads, d_model
        # C'est équivalent à faire: 
        # Mais reshape peut rendre le code plus court (en supprimant le besoin de .contiguous() après un transpose), 
        # mais elle masque la création potentielle d'une copie de données. 
        # Dans un code critique pour la performance comme le Transformer, 
        # beaucoup de développeurs préfèrent l'approche explicite de .view() et .contiguous() 
        # pour garder le contrôle sur les allocations de mémoire.
        # q = self.q_linear(q).reshape(q.size(0), -1, self.num_heads, self.d_k).transpose(1, 2) 
        # k = self.k_linear(k).reshape(k.size(0), -1, self.num_heads, self.d_k).transpose(1, 2) 
        # v = self.v_linear(v).reshape(v.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. Calcul de l'Attention (Scaled Dot-Product Attention)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k) # Shape: Seq Length, Seq Length

        # Application du masque (pour le Décodeur)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        scores = F.softmax(scores, dim=-1) # 

        # 3. Pondération des Valeurs (V)
        output = torch.matmul(scores, v)

        # 4. Concaténation et Couche de Sortie
        output = output.transpose(1, 2).contiguous().view(output.size(0), -1, self.d_model)
        output = self.out(output)

        # Le calcul est en O(Seq Len ** 2): on comprend l'intérêt des window size qu'il faut calculer au training
        return output

class PositionalEncoding(nn.Module): # Embedding Statique
    """Ajout de l'information de position aux embeddings"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x est l'embedding. On ajoute l'encodage positionnel
        x = x + self.pe[:, :x.size(1)]
        return x


class FeedForward(nn.Module):
    """Le réseau de neurones Feed-Forward à l'intérieur de l'Encodeur/Décodeur"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff) # d_ff souvent 4 * d_model
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    
class LayerNorm(nn.Module):
    """Implémentation manuelle de la Layer Normalization (appliquée sur la dernière dimension)"""
    def __init__(self, features, eps=1e-6):
        super().__init__()
        # Paramètres apprenables gamma (gain) et beta (biais/décalage)
        self.a_2 = nn.Parameter(torch.ones(features))  # gamma (initialisé à 1)
        self.b_2 = nn.Parameter(torch.zeros(features)) # beta (initialisé à 0)
        self.eps = eps

    def forward(self, x):
        # x: (Batch Size, Seq Length, d_model)
        
        # 1. Calcul de la moyenne et de la variance sur la dernière dimension (d_model)
        # La moyenne est calculée sur la dimension des features (-1)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) # std = sqrt(variance)

        # 2. Normalisation (Standardisation)
        x = (x - mean) / (std + self.eps)

        # 3. Mise à l'échelle et décalage (apprentissage)
        # Multiplication par gamma (a_2) et ajout de beta (b_2)
        return self.a_2 * x + self.b_2

# --- 2. Blocs Encodeur et Décodeur (avec Normalisation) ---

class EncoderLayer(nn.Module):
    """Représente un seul bloc de l'Encodeur"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 1. Multi-Head Self-Attention (avec Residual Connection)
        x_norm = self.norm1(x)
        attention_output = self.attn(x_norm, x_norm, x_norm, mask)
        x = x + self.dropout1(attention_output) # Add & Norm

        # 2. Feed-Forward (avec Residual Connection)
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + self.dropout2(ffn_output) # Add & Norm
        return x

class DecoderLayer(nn.Module):
    """Représente un seul bloc du Décodeur"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.attn1 = MultiHeadAttention(d_model, num_heads) # Masked Self-Attention
        self.attn2 = MultiHeadAttention(d_model, num_heads) # Encoder-Decoder Attention
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        # 1. Masked Multi-Head Self-Attention
        x_norm = self.norm1(x)
        # Q=x, K=x, V=x, application du masque de futur (tgt_mask)
        attention_output = self.attn1(x_norm, x_norm, x_norm, tgt_mask) 
        x = x + self.dropout1(attention_output)

        # 2. Encoder-Decoder Attention
        x_norm = self.norm2(x)
        # Q=x, K=memory(Encoder Output), V=memory (application du masque source, src_mask)
        attention_output = self.attn2(x_norm, memory, memory, src_mask) 
        x = x + self.dropout2(attention_output)

        # 3. Feed-Forward
        x_norm = self.norm3(x)
        ffn_output = self.ffn(x_norm)
        x = x + self.dropout3(ffn_output)
        return x

# --- 3. L'Architecture Complète du Transformer ---

class Encoder(nn.Module):
    """Empile les blocs Encodeur"""
    def __init__(self, d_model, num_heads, d_ff, dropout, num_layers, vocab_size):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model) # On peut utiliser register_buffer sinon (à regarder)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        x = self.embeddings(src) # Shape:  Batch Size, Seq Length, d_model
        x = self.pe(x) # Shape: Batch Size, Seq Length, d_model
        for layer in self.layers:
            x = layer(x, src_mask) 
        return self.norm(x) # Le prof en a pas parlé, bizarre

class Decoder(nn.Module):
    """Empile les blocs Décodeur"""
    def __init__(self, d_model, num_heads, d_ff, dropout, num_layers, vocab_size):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, src_mask, tgt_mask):
        x = self.embeddings(tgt)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    """Modèle Transformer complet (Encoder-Decoder)"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, d_ff=2048, num_layers=6, dropout=0.1):
        super().__init__()
        
        self.encoder = Encoder(d_model, num_heads, d_ff, dropout, num_layers, src_vocab_size)
        self.decoder = Decoder(d_model, num_heads, d_ff, dropout, num_layers, tgt_vocab_size)
        self.output_linear = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialisation de Xavier pour la stabilité
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # src: Entrée (Langue A), tgt: Sortie partielle (Langue B)
        # X: Batch Size x Max seq len
        
        # 1. Encodage : Obtention du 'memory' ou contexte
        memory = self.encoder(src, src_mask)
        # memory: (Batch Size, Max src seq len, d_model)
        
        # 2. Décodage : Utilisation du contexte pour générer la prédiction
        output = self.decoder(tgt, memory, src_mask, tgt_mask)
        # output: (Batch Size, Max tgt seq len, d_model)
        
        # 3. Couche Linéaire Finale pour la classification (probabilités de mots)
        output = self.output_linear(output)
        # output: (Batch Size, Max tgt seq len, tgt_vocab_size)
        
        return output

# --- Exemple d'Utilisation ---

if __name__ == "__main__":
    # Définition des paramètres
    SRC_VOCAB_SIZE = 1000  # Taille du vocabulaire source
    TGT_VOCAB_SIZE = 1000  # Taille du vocabulaire cible
    D_MODEL = 512
    NUM_HEADS = 8
    NUM_LAYERS = 6
    MAX_SEQ_LEN = 10

    # Création des données d'entrée factices
    batch_size = 32
    src_data = torch.randint(1, SRC_VOCAB_SIZE, (batch_size, MAX_SEQ_LEN)) # Entrée : séquence d'indices
    tgt_data = torch.randint(1, TGT_VOCAB_SIZE, (batch_size, MAX_SEQ_LEN)) # Cible : séquence d'indices (décalée)

    # Création du modèle
    model = Transformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, d_model=D_MODEL, num_heads=NUM_HEADS, num_layers=NUM_LAYERS)
    print(f"Modèle Transformer créé avec {NUM_LAYERS} couches.")
    # 
    
    # Forward Pass
    output_logits = model(src_data, tgt_data)
    
    # Vérification des dimensions
    print(f"Dimensions de sortie (Batch, Seq_len, Vocab_size): {output_logits.shape}")
    
    # Entraînement et perte (exemple)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # ignore_index=0 si PAD=0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Note : Pour l'entraînement, la cible doit être la séquence décalée
    # (par exemple, tgt_data[:, 1:] comme cible, et output_logits[:, :-1] comme prédictions)
    
    # Exemple de calcul de perte sur une étape (simplifié)
    target_labels = tgt_data[:, 1:].contiguous().view(-1)
    predictions = output_logits[:, :-1].contiguous().view(-1, TGT_VOCAB_SIZE)
    
    loss = criterion(predictions, target_labels)
    print(f"Loss calculée (Cross Entropy): {loss.item():.4f}")
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Étape d'optimisation effectuée.")
