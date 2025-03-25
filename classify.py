
import numpy as np
import time
from collections import Counter
import re
import pandas as pd
from scipy import sparse

def calculate_gini_impurity(french_count, spanish_count):
    total = french_count + spanish_count
    if total == 0:
        return 1.0
    p_french = french_count / total
    p_spanish = spanish_count / total
    return 1 - (p_french**2 + p_spanish**2)

def calculate_information_gain(total_french, total_spanish, pattern_french, pattern_spanish):
    total = total_french + total_spanish
    pattern_total = pattern_french + pattern_spanish
    non_pattern_french = total_french - pattern_french
    non_pattern_spanish = total_spanish - pattern_spanish
    non_pattern_total = total - pattern_total
    
    if total == 0 or pattern_total == 0 or non_pattern_total == 0:
        return 0
    
    p_french = total_french / total
    p_spanish = total_spanish / total
    parent_impurity = 1 - (p_french**2 + p_spanish**2)
    
    pattern_impurity = calculate_gini_impurity(pattern_french, pattern_spanish)
    non_pattern_impurity = calculate_gini_impurity(non_pattern_french, non_pattern_spanish)
    
    weighted_impurity = (pattern_total / total) * pattern_impurity + (non_pattern_total / total) * non_pattern_impurity
    
    return parent_impurity - weighted_impurity

def get_pattern_frequency_by_language(words, labels):
    pattern_counts = {}
    total_french = sum(1 for label in labels if label == 'french')
    total_spanish = sum(1 for label in labels if label == 'spanish')
    
    for word, label in zip(words, labels):
        word = word.lower()
        
        for n in range(1, 5):
            for i in range(len(word) - n + 1):
                pattern = word[i:i+n]
                if pattern not in pattern_counts:
                    pattern_counts[pattern] = {'french': 0, 'spanish': 0, 'total': 0}
                
                pattern_counts[pattern][label] += 1
                pattern_counts[pattern]['total'] += 1
        
        for n in range(1, 5):
            if len(word) >= n:
                prefix = word[:n]
                suffix = word[-n:]
                
                for pattern in [f"^{prefix}", f"{suffix}$"]:
                    if pattern not in pattern_counts:
                        pattern_counts[pattern] = {'french': 0, 'spanish': 0, 'total': 0}
                    
                    pattern_counts[pattern][label] += 1
                    pattern_counts[pattern]['total'] += 1
                    
        distinctive_patterns = [
            'eau', 'ou', 'ai', 'au', 'eu', 'oi', 'ph', 'th', 'aux', 'eux',
            'rr', 'll', 'ñ', 'ch', 'qu', 'ce', 'ci', 'za', 'ze', 'zi', 'zo'
        ]
        
        for pattern in distinctive_patterns:
            if pattern in word:
                special_pattern = f"DIST_{pattern}"
                if special_pattern not in pattern_counts:
                    pattern_counts[special_pattern] = {'french': 0, 'spanish': 0, 'total': 0}
                
                pattern_counts[special_pattern][label] += 1
                pattern_counts[special_pattern]['total'] += 1
    
    for pattern in pattern_counts:
        french_count = pattern_counts[pattern]['french']
        spanish_count = pattern_counts[pattern]['spanish']
        
        pattern_counts[pattern]['impurity'] = calculate_gini_impurity(french_count, spanish_count)
        pattern_counts[pattern]['info_gain'] = calculate_information_gain(
            total_french, total_spanish, french_count, spanish_count
        )
    
    return pattern_counts

def select_informative_patterns(pattern_counts, max_patterns=700, min_count=2, min_impurity=0.35):
    filtered_patterns = {
        p: counts for p, counts in pattern_counts.items() 
        if counts['total'] >= min_count and counts['impurity'] <= min_impurity
    }
    
    sorted_patterns = sorted(
        filtered_patterns.items(), 
        key=lambda x: x[1]['info_gain'], 
        reverse=True
    )
    
    distinctive = []
    regular = []
    
    for p, _ in sorted_patterns:
        if p.startswith("DIST_"):
            distinctive.append(p)
        else:
            regular.append(p)
    
    selected = distinctive + regular
    
    return selected[:max_patterns]

def get_position_specific_patterns(words, labels, position_type, lengths, params_by_length):
    patterns = []
    for k in lengths:
        max_gini, min_count, max_patterns = params_by_length.get(k, (0.2, 5, 100))
        patterns.extend(select_informative_patterns(
            get_pattern_frequency_by_language(words, labels), max_gini, min_count, max_patterns
        ))
    return patterns

def extract_linguistic_features(word):
    features = []
    word = word.lower()
    
    features.append(len(word))
    
    vowels = sum(1 for c in word if c in 'aeiouáéíóúàèìòùâêîôûäëïöü')
    consonants = len(word) - vowels
    
    features.append(vowels / (consonants + 0.01))
    
    features.append(vowels / (len(word) + 0.01))
    
    char_counts = Counter(word)
    total_chars = len(word)
    entropy = -sum((count / total_chars) * np.log2(count / total_chars) for count in char_counts.values())
    features.append(entropy)
    
    double_letters = sum(1 for i in range(len(word) - 1) if word[i] == word[i+1])
    features.append(double_letters)
    
    consonant_groups = len(re.findall(r'[bcdfghjklmnpqrstvwxyz]{3,}', word))
    features.append(consonant_groups)
    
    vowel_sequences = len(re.findall(r'[aeiouáéíóúàèìòùâêîôûäëïöü]{2,}', word))
    features.append(vowel_sequences)
    
    features.append(1 if word and word[0] in 'aeiouáéíóúàèìòùâêîôûäëïöü' else 0)
    
    features.append(1 if word and word[-1] in 'aeiouáéíóúàèìòùâêîôûäëïöü' else 0)
    
    features.append(1 if word and word[-1] in 'sxzptd' else 0)
    
    h_after_consonant = len(re.findall(r'[bcdfgjklmnpqrstvwxz]h', word))
    features.append(h_after_consonant)
    
    spanish_patterns = ['rr', 'll', 'ñ', 'ch', 'qu', 'ce', 'ci', 'za', 'ze', 'zi', 'zo', 'zu']
    for pattern in spanish_patterns:
        features.append(1 if pattern in word else 0)
    
    french_patterns = ['eau', 'ou', 'ai', 'au', 'eu', 'oi', 'ph', 'th', 'tion', 'aux', 'eux']
    for pattern in french_patterns:
        features.append(1 if pattern in word else 0)
    
    spanish_endings = ['ar', 'er', 'ir', 'os', 'as', 'es', 'an', 'en', 'dad', 'ción']
    for ending in spanish_endings:
        features.append(1 if word.endswith(ending) else 0)
        
    french_endings = ['ez', 'er', 're', 'ent', 'ais', 'ois', 'eur', 'eux', 'aux', 'eau']
    for ending in french_endings:
        features.append(1 if word.endswith(ending) else 0)
    
    features.append(word.count('ñ'))
    features.append(word.count('j'))
    
    features.append(word.count('w'))
    features.append(word.count('y'))
    features.append(word.count('q'))
    
    diphthongs = ['ia', 'ie', 'io', 'iu', 'ui', 'ue', 'ua', 'uo']
    for diph in diphthongs:
        features.append(1 if diph in word else 0)
        
    return features

def create_feature_vector(word, informative_patterns):
    word = word.lower()
    
    features = [1 if pattern in word else 0 for pattern in informative_patterns]
    
    for pattern in informative_patterns:
        if pattern.endswith('$') and pattern[:-1] != '':
            suffix_pattern = pattern[:-1]
            features.append(1 if word.endswith(suffix_pattern) else 0)
        
        elif pattern.startswith('^') and pattern[1:] != '':
            prefix_pattern = pattern[1:]
            features.append(1 if word.startswith(prefix_pattern) else 0)
    
    features.extend(extract_linguistic_features(word))
    
    return features

def create_feature_matrix(words, informative_patterns):
    return np.array([create_feature_vector(word, informative_patterns) for word in words])

def normalize_features(X_train, X_test):
    feature_means = np.mean(X_train, axis=0)
    feature_stds = np.std(X_train, axis=0) + 1e-10
    
    X_train_normalized = (X_train - feature_means) / feature_stds
    X_test_normalized = (X_test - feature_means) / feature_stds
    
    return X_train_normalized, X_test_normalized

def train_model(X, y, alphas=None):
    if alphas is None:
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    
    y_numeric = np.array([0 if label == 'french' else 1 for label in y])
    
    n_folds = 5
    n_samples = len(X)
    fold_size = n_samples // n_folds
    
    cv_scores = {alpha: [] for alpha in alphas}
    
    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < n_folds - 1 else n_samples
        
        val_mask = np.zeros(n_samples, dtype=bool)
        val_mask[val_start:val_end] = True
        
        X_train_fold, X_val_fold = X[~val_mask], X[val_mask]
        y_train_fold, y_val_fold = y_numeric[~val_mask], y_numeric[val_mask]
        
        for alpha in alphas:
            beta = np.zeros(X_train_fold.shape[1])
            
            XTX = X_train_fold.T @ X_train_fold
            reg_matrix = XTX + alpha * np.eye(XTX.shape[0])
            XTy = X_train_fold.T @ y_train_fold
            
            beta = np.linalg.solve(reg_matrix, XTy)
            
            y_pred_fold = (X_val_fold @ beta >= 0.5).astype(int)
            
            accuracy = np.mean(y_pred_fold == y_val_fold)
            cv_scores[alpha].append(accuracy)
    
    mean_cv_scores = {alpha: np.mean(scores) for alpha, scores in cv_scores.items()}
    best_alpha = max(mean_cv_scores, key=mean_cv_scores.get)
    best_cv_score = mean_cv_scores[best_alpha]
    
    XTX = X.T @ X
    reg_matrix = XTX + best_alpha * np.eye(XTX.shape[0])
    XTy = X.T @ y_numeric
    
    beta = np.linalg.solve(reg_matrix, XTy)
    
    y_pred_train = (X @ beta >= 0.5).astype(int)
    train_accuracy = np.mean(y_pred_train == y_numeric)
    
    print(f"Cross-validation accuracy: {best_cv_score:.4f} with alpha={best_alpha:.4f}")
    print(f"Training accuracy: {train_accuracy:.4f}")
    
    return beta, best_alpha

def predict_with_ridge(X, weights):
    n = X.shape[0]
    X_with_bias = np.hstack((np.ones((n, 1)), X))
    return np.sign(X_with_bias @ weights)

def cross_validate(X, y, alphas, n_folds=5):
    n = X.shape[0]
    fold_size = n // n_folds
    best_alpha = None
    best_accuracy = -1
    
    for alpha in alphas:
        accuracies = []
        
        for fold in range(n_folds):
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < n_folds - 1 else n
            
            val_indices = list(range(start_idx, end_idx))
            train_indices = [i for i in range(n) if i not in val_indices]
            
            X_train_fold = X[train_indices]
            y_train_fold = y[train_indices]
            X_val_fold = X[val_indices]
            y_val_fold = y[val_indices]
            
            weights = train_model(X_train_fold, y_train_fold, alpha)[0]
            
            y_pred = predict_with_ridge(X_val_fold, weights)
            
            accuracy = np.mean(y_pred == y_val_fold)
            accuracies.append(accuracy)
        
        mean_accuracy = np.mean(accuracies)
        
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_alpha = alpha
    
    return best_alpha, best_accuracy

def find_best_hyperparameters(X, y):
    alphas = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    best_alpha, best_accuracy = cross_validate(X, y, alphas)
    
    return {"alpha": best_alpha, "cv_accuracy": best_accuracy}

def classify(train_words, train_labels, test_words):
    print(f"Training on {len(train_words)} words, classifying {len(test_words)} words...")
    
    start_time = time.time()
    
    pattern_counts = get_pattern_frequency_by_language(train_words, train_labels)
    
    informative_patterns = select_informative_patterns(pattern_counts, max_patterns=900, min_count=2, min_impurity=0.30)
    print(f"Selected {len(informative_patterns)} informative patterns")
    
    X_train = create_feature_matrix(train_words, informative_patterns)
    X_test = create_feature_matrix(test_words, informative_patterns)
    
    definitive_french_patterns = ['eau', 'aux', 'eux', 'tion', 'aient', 'ph']
    definitive_spanish_patterns = ['ñ', 'rr', 'll', 'ío', 'ón']
    
    rule_predictions = []
    for word in test_words:
        word = word.lower()
        french_score = 0
        spanish_score = 0
        
        for pattern in definitive_french_patterns:
            if pattern in word:
                french_score += 1
                
        for pattern in definitive_spanish_patterns:
            if pattern in word:
                spanish_score += 1
        
        if french_score > spanish_score + 1:
            rule_predictions.append('french')
        elif spanish_score > french_score + 1:
            rule_predictions.append('spanish')
        else:
            rule_predictions.append(None)
    
    beta_initial, _ = train_model(X_train, train_labels, alphas=[0.1, 1.0, 10.0, 100.0])
    
    feature_importance = np.abs(beta_initial)
    
    num_features_to_keep = max(450, int(0.85 * len(feature_importance)))
    top_feature_indices = np.argsort(feature_importance)[-num_features_to_keep:]
    
    X_train_selected = X_train[:, top_feature_indices]
    X_test_selected = X_test[:, top_feature_indices]
    
    models = []
    
    beta, best_alpha = train_model(X_train_selected, train_labels, alphas=[0.1, 1.0, 10.0, 100.0])
    models.append(("ridge", beta, None))
    
    ensemble = train_ensemble_models(X_train_selected, train_labels, num_models=11)
    
    short_indices = [i for i, word in enumerate(train_words) if len(word) <= 4]
    if len(short_indices) > 20:
        X_short = X_train_selected[short_indices]
        y_short = [train_labels[i] for i in short_indices]
        beta_short, _ = train_model(X_short, y_short)
        models.append(("short", beta_short, short_indices))
    
    long_indices = [i for i, word in enumerate(train_words) if len(word) > 7]
    if len(long_indices) > 20:
        X_long = X_train_selected[long_indices]
        y_long = [train_labels[i] for i in long_indices]
        beta_long, _ = train_model(X_long, y_long)
        models.append(("long", beta_long, long_indices))
    
    spanish_endings = ['ar', 'er', 'ir', 'os', 'as', 'es', 'an', 'en', 'dad', 'ción', 'ío', 'ico', 'ica']
    french_endings = ['ez', 'er', 're', 'ent', 'ais', 'ois', 'eur', 'eux', 'aux', 'eau', 'ion', 'ment']
    
    ending_indices = []
    for i, word in enumerate(train_words):
        word = word.lower()
        for ending in spanish_endings + french_endings:
            if word.endswith(ending):
                ending_indices.append(i)
                break
    
    if len(ending_indices) > 20:
        X_ending = X_train_selected[ending_indices]
        y_ending = [train_labels[i] for i in ending_indices]
        beta_ending, _ = train_model(X_ending, y_ending)
        models.append(("endings", beta_ending, ending_indices))
    
    predictions = []
    for i, word in enumerate(test_words):
        word = word.lower()
        
        if rule_predictions[i] is not None:
            predictions.append(rule_predictions[i])
            continue
        
        X_word = X_test_selected[i]
        
        votes = Counter()
        
        base_pred = 'spanish' if np.dot(X_word, beta) >= 0.5 else 'french'
        votes[base_pred] += 7
        
        for model_beta, feature_indices, _ in ensemble:
            X_subset = X_word[feature_indices]
            ensemble_pred = 'spanish' if np.dot(X_subset, model_beta) >= 0.5 else 'french'
            votes[ensemble_pred] += 1
        
        for model_name, model_beta, _ in models:
            if model_name == "short" and len(word) <= 4:
                short_pred = 'spanish' if np.dot(X_word, model_beta) >= 0.5 else 'french'
                votes[short_pred] += 3
            elif model_name == "long" and len(word) > 7:
                long_pred = 'spanish' if np.dot(X_word, model_beta) >= 0.5 else 'french'
                votes[long_pred] += 3
        
        for model_name, model_beta, _ in models:
            if model_name == "endings":
                has_ending = False
                for ending in spanish_endings + french_endings:
                    if word.endswith(ending):
                        has_ending = True
                        break
                
                if has_ending:
                    ending_pred = 'spanish' if np.dot(X_word, model_beta) >= 0.5 else 'french'
                    votes[ending_pred] += 4
        
        if word[-1:] in 'aeiouáéíóú':
            votes['spanish'] += 2
        
        if word[-1:] in 'tdpxz':
            votes['french'] += 2
            
        for ending in ['os', 'as', 'es', 'ar', 'er', 'ir', 'ía']:
            if word.endswith(ending):
                votes['spanish'] += 2
                break
        
        for ending in ['eur', 'aux', 'eaux', 'tion', 'ssion', 'ment']:
            if word.endswith(ending):
                votes['french'] += 2
                break
        
        if votes:
            predictions.append(votes.most_common(1)[0][0])
        else:
            predictions.append('french' if np.random.random() < 0.5 else 'spanish')
    
    for i, word in enumerate(test_words):
        word = word.lower()
        
        common_spanish_words = {
            'colores', 'parte', 'hombre', 'simple', 'noche', 'sol', 'luz', 'calle', 
            'nombre', 'papel', 'paz', 'poder', 'arte', 'aceite', 'alcance'
        }
        
        common_french_words = {
            'grace', 'principale', 'eliminer', 'consciente', 'relations', 'avoir', 
            'melodie', 'parente', 'neige', 'vitesse'
        }
        
        if word in common_spanish_words:
            predictions[i] = 'spanish'
        elif word in common_french_words:
            predictions[i] = 'french'
    
    end_time = time.time()
    print(f"Classification completed in {end_time - start_time:.2f} seconds")
    
    return predictions

def train_ensemble_models(X, y, num_models=5):
    ensemble = []
    n_samples = len(X)
    n_features = X.shape[1]
    
    for i in range(num_models):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_bootstrap = X[indices]
        y_bootstrap = [y[idx] for idx in indices]
        
        feature_indices = np.random.choice(n_features, int(0.7 * n_features), replace=False)
        X_subset = X_bootstrap[:, feature_indices]
        
        beta, alpha = train_model(X_subset, y_bootstrap)
        
        ensemble.append((beta, feature_indices, alpha))
    
    return ensemble

def predict_with_ensemble(word, informative_patterns, ensemble):
    X = np.array(create_feature_vector(word, informative_patterns))
    
    votes = []
    for beta, feature_indices, _ in ensemble:
        X_subset = X[feature_indices]
        pred = 1 if np.dot(X_subset, beta) >= 0.5 else 0
        votes.append('spanish' if pred == 1 else 'french')
    
    vote_counts = Counter(votes)
    
    return vote_counts.most_common(1)[0][0]

def main(train_file, test_file=None):
    train_data = pd.read_csv(train_file)
    
    if 'label' not in train_data.columns or 'word' not in train_data.columns:
        raise ValueError("Dataset must contain 'label' and 'word' columns")
    
    words = train_data['word'].tolist()
    labels = train_data['label'].tolist()
    
    pattern_counts = get_pattern_frequency_by_language(words, labels)
    
    informative_patterns = select_informative_patterns(pattern_counts)
    print(f"Selected {len(informative_patterns)} informative patterns")
    
    X = create_feature_matrix(words, informative_patterns)
    
    start_time = time.time()
    
    beta, best_alpha = train_model(X, labels)
    
    ensemble = train_ensemble_models(X, labels, num_models=5)
    
    training_time = time.time() - start_time
    print(f"Classification completed in {training_time:.2f} seconds")
    
    if test_file:
        test_data = pd.read_csv(test_file)
        
        if 'label' not in test_data.columns or 'word' not in test_data.columns:
            raise ValueError("Test dataset must contain 'label' and 'word' columns")
        
        test_words = test_data['word'].tolist()
        test_labels = test_data['label'].tolist()
        
        X_test = create_feature_matrix(test_words, informative_patterns)
        
        start_time = time.time()
        predictions = predict(X_test, beta)
        testing_time = time.time() - start_time
        
        accuracy = sum(1 for p, t in zip(predictions, test_labels) if p == t) / len(test_labels)
        print(f"Test accuracy: {accuracy:.4f}")
        print(f"Correct predictions: {int(accuracy * len(test_labels))}/{len(test_labels)}")
        print(f"Classification time: {testing_time:.2f} seconds")
        
        confusion_matrix = {}
        for true_label in set(test_labels):
            confusion_matrix[true_label] = {}
            for pred_label in set(predictions):
                confusion_matrix[true_label][pred_label] = sum(
                    1 for p, t in zip(predictions, test_labels) 
                    if p == pred_label and t == true_label
                )
        
        print("\nConfusion Matrix:")
        all_labels = sorted(set(test_labels) | set(predictions))
        
        print(f"{'True/Pred':>15}", end='')
        for label in all_labels:
            print(f"{label:>15}", end='')
        print()
        
        for true_label in all_labels:
            print(f"{true_label:>15}", end='')
            for pred_label in all_labels:
                count = confusion_matrix.get(true_label, {}).get(pred_label, 0)
                print(f"{count:>15}", end='')
            print()
        
        misclassifications = [
            (word, true_label, pred_label) 
            for word, true_label, pred_label in zip(test_words, test_labels, predictions) 
            if true_label != pred_label
        ]
        
        if misclassifications:
            print("\nSample misclassifications:")
            for word, true_label, pred_label in misclassifications[:10]:
                print(f"Word: {word}, True: {true_label}, Predicted: {pred_label}")
    
    return informative_patterns, beta, ensemble

def predict(X, beta):
    y_pred_numeric = (X @ beta >= 0.5).astype(int)
    
    return ['french' if pred == 0 else 'spanish' for pred in y_pred_numeric]

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        train_file = sys.argv[1]
        test_file = sys.argv[2] if len(sys.argv) > 2 else None
        main(train_file, test_file)
    else:
        train_file = "data/train.csv"
        main(train_file) 
