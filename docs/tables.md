### Official test split · seq_auc

| Model | All frames: bounding box | All frames: pose | All frames: bounding box + pose | Before decision point: bounding box | Before decision point: pose | Before decision point: bounding box + pose |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.833 | 0.941 | 0.960 | 0.723 | 0.915 | 0.807 |
| Support Vector Machine | 0.927 | **0.955** | 0.967 | **0.931** | **0.947** | **0.830** |
| Random Forest | 0.931 | 0.915 | 0.961 | 0.741 | 0.718 | 0.661 |
| Gradient Boosting | 0.906 | 0.895 | **0.969** | 0.743 | 0.669 | 0.594 |
| Long Short-Term Memory | 0.904 | 0.915 | 0.940 | 0.715 | 0.632 | 0.615 |
| Gated Recurrent Unit | 0.938 | 0.929 | 0.949 | 0.706 | 0.501 | 0.667 |
| Temporal Convolutional Network | **0.944** | 0.922 | 0.961 | 0.835 | 0.914 | 0.769 |
| Transformer | 0.931 | 0.935 | 0.948 | 0.794 | 0.492 | 0.774 |

### Official test split · seq_bacc

| Model | All frames: bounding box | All frames: pose | All frames: bounding box + pose | Before decision point: bounding box | Before decision point: pose | Before decision point: bounding box + pose |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.748 | 0.849 | 0.862 | 0.792 | 0.699 | **0.819** |
| Support Vector Machine | 0.831 | **0.877** | **0.910** | 0.814 | 0.803 | 0.803 |
| Random Forest | 0.854 | 0.827 | 0.898 | 0.739 | 0.492 | 0.497 |
| Gradient Boosting | 0.847 | 0.820 | 0.852 | 0.713 | 0.632 | 0.561 |
| Long Short-Term Memory | 0.834 | 0.825 | 0.867 | 0.672 | 0.539 | 0.520 |
| Gated Recurrent Unit | 0.860 | 0.838 | 0.865 | 0.676 | 0.413 | 0.664 |
| Temporal Convolutional Network | 0.855 | 0.831 | 0.895 | **0.827** | **0.815** | 0.684 |
| Transformer | **0.871** | 0.834 | 0.868 | 0.708 | 0.339 | 0.656 |

### Official test split · ped_auc

| Model | All frames: bounding box | All frames: pose | All frames: bounding box + pose | Before decision point: bounding box | Before decision point: pose | Before decision point: bounding box + pose |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.962 | 0.980 | 0.994 | 0.787 | **0.933** | **0.920** |
| Support Vector Machine | 0.991 | **0.994** | 0.994 | **0.920** | 0.920 | 0.893 |
| Random Forest | **0.993** | 0.942 | 0.989 | 0.720 | 0.724 | 0.702 |
| Gradient Boosting | 0.980 | 0.949 | 0.993 | 0.787 | 0.667 | 0.613 |
| Long Short-Term Memory | 0.992 | 0.941 | 0.989 | 0.733 | 0.640 | 0.747 |
| Gated Recurrent Unit | 0.991 | 0.953 | 0.988 | 0.716 | 0.507 | 0.729 |
| Temporal Convolutional Network | 0.990 | 0.957 | **0.998** | 0.889 | 0.920 | 0.884 |
| Transformer | 0.978 | 0.949 | 0.981 | 0.733 | 0.538 | 0.880 |

### 5-fold CV · seq_auc

| Model | All frames: bounding box | All frames: pose | All frames: bounding box + pose | Before decision point: bounding box | Before decision point: pose | Before decision point: bounding box + pose |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.877 | 0.857 | 0.890 | 0.843 | 0.624 | 0.791 |
| Support Vector Machine | 0.888 | 0.865 | 0.895 | **0.893** | 0.692 | 0.770 |
| Random Forest | **0.901** | **0.873** | 0.904 | 0.823 | 0.680 | 0.755 |
| Gradient Boosting | 0.898 | 0.859 | **0.920** | 0.796 | 0.640 | 0.786 |
| Long Short-Term Memory | 0.856 | 0.828 | 0.844 | 0.847 | 0.686 | 0.775 |
| Gated Recurrent Unit | 0.893 | 0.832 | 0.861 | 0.841 | 0.704 | **0.835** |
| Temporal Convolutional Network | 0.886 | 0.826 | 0.877 | 0.871 | **0.711** | 0.803 |
| Transformer | 0.883 | 0.824 | 0.868 | 0.871 | 0.643 | 0.807 |

### 5-fold CV · seq_bacc

| Model | All frames: bounding box | All frames: pose | All frames: bounding box + pose | Before decision point: bounding box | Before decision point: pose | Before decision point: bounding box + pose |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.798 | **0.787** | 0.800 | 0.705 | 0.518 | 0.657 |
| Support Vector Machine | 0.812 | 0.764 | 0.790 | 0.747 | 0.582 | 0.606 |
| Random Forest | 0.813 | 0.784 | 0.814 | 0.702 | 0.579 | 0.640 |
| Gradient Boosting | **0.814** | 0.763 | **0.823** | 0.707 | 0.578 | 0.713 |
| Long Short-Term Memory | 0.777 | 0.721 | 0.739 | 0.685 | 0.604 | 0.704 |
| Gated Recurrent Unit | 0.794 | 0.739 | 0.756 | 0.756 | 0.624 | **0.736** |
| Temporal Convolutional Network | 0.783 | 0.720 | 0.764 | **0.778** | **0.625** | 0.731 |
| Transformer | 0.793 | 0.732 | 0.754 | 0.763 | 0.562 | 0.723 |

### 5-fold CV · ped_auc

| Model | All frames: bounding box | All frames: pose | All frames: bounding box + pose | Before decision point: bounding box | Before decision point: pose | Before decision point: bounding box + pose |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.946 | 0.925 | 0.949 | 0.848 | 0.631 | 0.799 |
| Support Vector Machine | 0.957 | **0.945** | 0.964 | **0.888** | 0.678 | 0.755 |
| Random Forest | 0.962 | 0.927 | 0.958 | 0.805 | 0.659 | 0.734 |
| Gradient Boosting | **0.964** | 0.926 | **0.972** | 0.785 | 0.629 | 0.786 |
| Long Short-Term Memory | 0.919 | 0.891 | 0.911 | 0.831 | 0.674 | 0.757 |
| Gated Recurrent Unit | 0.943 | 0.877 | 0.915 | 0.824 | 0.687 | **0.821** |
| Temporal Convolutional Network | 0.954 | 0.885 | 0.948 | 0.869 | **0.705** | 0.787 |
| Transformer | 0.936 | 0.865 | 0.921 | 0.844 | 0.643 | 0.802 |
