import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from transformers import BertTokenizer, BertModel
from wordcloud import WordCloud
# Import components from your dl_base file.
from dl_base import GenericModel, BCELoss, TransformerInputLayer

###########################################
# Dummy Args class for GenericModel
###########################################
class DummyArgs:
    world_size = 1
    fp32encoder = True  
    fp16xfc = False  
    accum = 1         
    noloss = False    
    checkpoint_resume = ""  
    custom_cuda = False    
    default_impl = True   
    bottleneck_dims = 768  

###########################################
# Data Loading Functions
###########################################
def parse_trn_X_Y_file(filepath):
    mappings = []
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    header_parts = lines[0].strip().split()
    if len(header_parts) == 2 and header_parts[0].isdigit() and header_parts[1].isdigit():
        lines = lines[1:]
    for line in lines:
        line = line.strip()
        if not line:
            mappings.append([])
            continue
        parts = line.split()
        label_score_pairs = []
        for part in parts:
            if ':' in part:
                lbl_str, score_str = part.split(':', 1)
                label_id = int(lbl_str)
                score = float(score_str)
            else:
                label_id = int(part)
                score = 1.0
            label_score_pairs.append((label_id, score))
        mappings.append(label_score_pairs)
    return mappings

def load_instance_with_labels(dataset_dir, instance_index=None):
    trn_text_file = os.path.join(dataset_dir, "trn_X.txt")
    trn_mapping_file = os.path.join(dataset_dir, "trn_X_Y.txt")
    labels_file = os.path.join(dataset_dir, "Y.txt")
    
    with open(trn_text_file, "r", encoding="utf-8") as f:
        texts = f.readlines()
    
    mappings = parse_trn_X_Y_file(trn_mapping_file)
    
    with open(labels_file, "r", encoding="utf-8") as f:
        label_texts = [line.strip() for line in f.readlines()]
    
    if instance_index is None:
        instance_index = random.randint(0, len(texts) - 1)
    
    instance_text = texts[instance_index].strip()
    instance_label_ids = [label_id for (label_id, score) in mappings[instance_index]]
    
    return instance_text, instance_label_ids, label_texts

###########################################
# RENEE Inference Wrapper
###########################################
class RENEEWrapper(torch.nn.Module):
    def __init__(self, loss_model):
        super(RENEEWrapper, self).__init__()
        self.loss_model = loss_model
        
    def forward(self, **inputs):
        batch_data = {
            'xfts': {
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask']
            }
        }
        embed = self.loss_model(batch_data)
        logits = torch.matmul(embed, self.loss_model.model.xfc_weight.t())
        class OutWrapper:
            def __init__(self, logits):
                self.logits = logits
        return OutWrapper(logits)

###########################################
# LIME Explainer
###########################################
class LIMEExplainer:
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def get_interpretable_text(self, text):
        tokens = self.tokenizer.tokenize(text)
        return tokens

    def perturb_instance(self, tokens, num_samples=500, perturbation_rate=0.3):
        perturbations = []
        binary_masks = []
        for _ in range(num_samples):
            mask = [1 if random.random() > perturbation_rate else 0 for _ in tokens]
            if sum(mask) == 0:
                mask[random.randint(0, len(tokens)-1)] = 1
            binary_masks.append(mask)
            perturbed_tokens = [tok for tok, m in zip(tokens, mask) if m == 1]
            perturbations.append(perturbed_tokens)
        return perturbations, np.array(binary_masks)

    def reconstruct_text(self, tokens):
        return self.tokenizer.convert_tokens_to_string(tokens)

    def predict(self, texts, target_label=None, max_length=128):
        self.model.eval()
        preds = []
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt",
                                        max_length=max_length, truncation=True, padding="max_length")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                probs = torch.sigmoid(logits).cpu().numpy()[0]
                if target_label is not None:
                    preds.append(probs[target_label])
                else:
                    preds.append(probs)
        return np.array(preds)

    def explain_instance(self, text, target_label, num_samples=500, perturbation_rate=0.3):
        tokens = self.get_interpretable_text(text)
        perturbations, binary_masks = self.perturb_instance(tokens, num_samples=num_samples, perturbation_rate=perturbation_rate)
        perturbed_texts = [self.reconstruct_text(p) for p in perturbations]
        predictions = self.predict(perturbed_texts, target_label=target_label)
        original_mask = np.ones(len(tokens))
        distances = np.mean(np.abs(binary_masks - original_mask), axis=1)
        sigma = np.sqrt(len(tokens)) * 0.25
        weights = np.exp(- (distances ** 2) / (sigma ** 2))
        reg = LinearRegression()
        reg.fit(binary_masks, predictions, sample_weight=weights)
        return tokens, reg.coef_, reg.intercept_, binary_masks, predictions, distances, weights

###########################################
# Utility Functions for Aligning Coefficients
###########################################
def align_coefficients(tokens1, coefs1, tokens2, coefs2):
    """
    Aligns two sets of token coefficients based on the union of tokens.
    Missing tokens are assigned a coefficient of 0.
    Returns two numpy vectors of aligned coefficients.
    """
    d1 = dict(zip(tokens1, coefs1))
    d2 = dict(zip(tokens2, coefs2))
    all_tokens = set(tokens1) | set(tokens2)
    v1 = np.array([d1.get(token, 0.0) for token in all_tokens])
    v2 = np.array([d2.get(token, 0.0) for token in all_tokens])
    return v1, v2

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

###########################################
# Visualization Functions
###########################################
def plot_bar_chart(tokens, coefs, title="Token Importance", save_path=None):
    token_importances = list(zip(tokens, coefs))
    token_importances = sorted(token_importances, key=lambda x: abs(x[1]), reverse=True)
    tokens_sorted, coefs_sorted = zip(*token_importances)
    
    plt.figure(figsize=(10, 6))
    plt.barh(tokens_sorted, coefs_sorted, color="skyblue")
    plt.xlabel("Coefficient Value")
    plt.title(title)
    plt.gca().invert_yaxis()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_word_cloud(tokens, coefs, title="Word Cloud of Token Importances", save_path=None):
    token_dict = {token: abs(coef) for token, coef in zip(tokens, coefs)}
    wc = WordCloud(width=800, height=400, background_color="white")
    wc.generate_from_frequencies(token_dict)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.title(title)
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_heatmap(instance_labels, explanation_matrix, tokens=None, title="Heatmap of Token Importances",save_path=None):
    if tokens is None:
        tokens = [f"Token {i}" for i in range(explanation_matrix.shape[1])]
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(explanation_matrix, annot=True, cmap="coolwarm", xticklabels=tokens, yticklabels=instance_labels)
    plt.title(title)
    plt.xlabel("Tokens")
    plt.ylabel("Instances")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_consistency(original_tokens, orig_coefs, pert_tokens, pert_coefs, title="Consistency: Original vs Perturbed",save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    orig_importances = list(zip(original_tokens, orig_coefs))
    orig_importances = sorted(orig_importances, key=lambda x: abs(x[1]), reverse=True)
    tokens_orig, coefs_orig = zip(*orig_importances)
    axes[0].barh(tokens_orig, coefs_orig, color="skyblue")
    axes[0].set_title("Original Instance")
    axes[0].invert_yaxis()
    
    pert_importances = list(zip(pert_tokens, pert_coefs))
    pert_importances = sorted(pert_importances, key=lambda x: abs(x[1]), reverse=True)
    tokens_pert, coefs_pert = zip(*pert_importances)
    axes[1].barh(tokens_pert, coefs_pert, color="salmon")
    axes[1].set_title("Perturbed Instance")
    axes[1].invert_yaxis()
    
    plt.suptitle(title)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def log_explanation(instance_text, target_label, label_text, tokens, coefs, intercept,consistency_score, file_path="lime_explanation_log.txt"):
    with open(file_path, "a", encoding="utf-8") as f:
        f.write("Instance: {}\n".format(instance_text))
        f.write("Explained Label: {} (index {})\n".format(label_text, target_label))
        f.write("Tokens and Importance:\n")
        for token, coef in zip(tokens, coefs):
            f.write("{:<15s}: {:.4f}\n".format(token, coef))
        f.write("Intercept: {:.4f}\n".format(intercept))
        f.write("---\n")
        f.write("\nCosine Similarity between original and perturbed explanations: {:.4f}\n".format(consistency_score))
        f.write("------------------------------------------------------------------------------------------------\n")
        # print("\nCosine Similarity between original and perturbed explanations: {:.4f}".format(consistency_score))
        

###########################################
# Main Script: Load Data, Build Model, Run LIME & Visualizations
###########################################
if __name__ == "__main__":
    dataset_directory = "Datasets/LF-AmazonTitles-131K"  # Adjust as needed.
    
    instance_text, instance_label_ids, all_label_texts = load_instance_with_labels(dataset_directory)
    print("Selected Instance Text:")
    print(instance_text)
    print("\nAssociated Label IDs:", instance_label_ids)
    print("Associated Label Texts:", [all_label_texts[i] for i in instance_label_ids])
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    num_labels = len(all_label_texts)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ###########################################
    # Build the RENEE Model for Inference
    ###########################################
    args = DummyArgs()
    transformer = BertModel.from_pretrained("bert-base-uncased")
    input_layer = TransformerInputLayer(transformer)
    
    from collections import OrderedDict
    embed_modules = OrderedDict([("pooler", input_layer)])
    
    rank = 0
    per_gpu_batch_size = 1
    numy = num_labels
    numy_per_gpu = num_labels
    out_dir = "./model_out"
    
    model_arch = GenericModel(rank, args, numy, numy_per_gpu, per_gpu_batch_size,
                              modules=embed_modules, device=device, name="renee", out_dir=out_dir)
    loss_model = BCELoss(model_arch)
    renee_model = RENEEWrapper(loss_model).to(device)
    renee_model.eval()
    
    ###########################################
    # LIME Explanation for a Selected Instance
    ###########################################
    explainer = LIMEExplainer(renee_model, tokenizer, device=device)
    target_label = instance_label_ids[0]
    print("\nGenerating LIME explanation for label index:", target_label)
    print("Label Name:", all_label_texts[target_label])
    
    tokens, coefs, intercept, binary_masks, predictions, distances, weights = explainer.explain_instance(
                                        instance_text, target_label=target_label,
                                        num_samples=500, perturbation_rate=0.3)
    
    
    
    print("\nLIME Explanation (Token Importances):")
    for token, coef in zip(tokens, coefs):
        print("{:<15s}: {:.4f}".format(token, coef))
    
    ###########################################
    # Visualization: Horizontal Bar Chart
    ###########################################
    plot_bar_chart(tokens, coefs, title="LIME Explanation for Label '{}'".format(all_label_texts[target_label]), 
                   save_path="lime_bar.png")
    
    ###########################################
    # Visualization: Word Cloud
    ###########################################
    plot_word_cloud(tokens, coefs, title="Word Cloud of Token Importances", save_path="lime_wc.png")
    
    ###########################################
    # Consistency Check & Visualization: Side-by-Side Comparison
    ###########################################
    perturbed_instance_text = instance_text + " ?"
    tokens_pert, coefs_pert, intercept_pert, _, _, _, _ = explainer.explain_instance(
                                perturbed_instance_text, target_label=target_label,
                                num_samples=500, perturbation_rate=0.3)
    
    # Align coefficient vectors based on token names
    aligned_coefs_orig, aligned_coefs_pert = align_coefficients(tokens, coefs, tokens_pert, coefs_pert)
    consistency_score = cosine_similarity(aligned_coefs_orig, aligned_coefs_pert)
    print("\nCosine Similarity between original and perturbed explanations: {:.4f}".format(consistency_score))
    
    log_explanation(instance_text, target_label, all_label_texts[target_label], tokens, coefs, intercept, consistency_score)
    
    plot_consistency(tokens, coefs, tokens_pert, coefs_pert, title="Consistency: Original vs Perturbed",save_path="lime_consistency.png")
    
    ###########################################
    # Visualization: Heatmap Across Instances
    ###########################################
    # For demonstration, we generate explanations for two instances (original and perturbed) using the same token alignment.
    # First, align coefficients and also sort tokens for a uniform display.
    aligned_tokens = list(set(tokens) | set(tokens_pert))
    # Create aligned coefficient vectors for both instances.
    d_orig = dict(zip(tokens, coefs))
    d_pert = dict(zip(tokens_pert, coefs_pert))
    aligned_coefs_orig = np.array([d_orig.get(tok, 0.0) for tok in aligned_tokens])
    aligned_coefs_pert = np.array([d_pert.get(tok, 0.0) for tok in aligned_tokens])
    
    explanation_matrix = np.vstack([aligned_coefs_orig, aligned_coefs_pert])
    instance_labels = ["Original", "Perturbed"]
    plot_heatmap(instance_labels, explanation_matrix, tokens=aligned_tokens, 
                 title="Heatmap of Token Importances: Original vs Perturbed",save_path="lime_heatmap.png")
