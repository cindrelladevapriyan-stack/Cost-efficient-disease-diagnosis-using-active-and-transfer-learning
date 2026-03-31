Cost-Efficient Disease Diagnosis using Active & Transfer Learning
Overview
This project presents a cost-efficient disease diagnosis system that combines Active Learning and Transfer Learning to reduce dependency on large labeled datasets while maintaining effective predictive performance.
In healthcare, labeled data is expensive and requires expert annotation. This project addresses that challenge by selectively choosing the most informative samples for labeling, significantly reducing cost while still achieving meaningful model performance.
Objectives
Reduce the cost of data labeling in medical diagnosis
Improve model efficiency with limited labeled data
Utilize pre-trained models for better feature extraction
Build a scalable and practical ML pipeline
Methodology
Transfer Learning
Leveraged pre-trained models to extract meaningful features
Fine-tuned the model on a domain-specific dataset
Reduced training time and improved generalization
Active Learning
Implemented an iterative training loop
Selected the most uncertain/informative samples for labeling
Reduced the number of labeled samples required
Workflow
Data preprocessing and cleaning
Feature extraction using transfer learning
Active learning sample selection
Model training and evaluation
Iterative improvement
Tech Stack
Python
Scikit-learn
TensorFlow / PyTorch (update based on your usage)
NumPy, Pandas
Matplotlib / Seaborn
Results
Macro F1 Score: 0.1908
AUC Score: 0.7367
The model demonstrates strong discriminative ability (AUC) despite being trained on limited labeled data, highlighting the effectiveness of combining active and transfer learning for cost-efficient diagnosis systems.
Project Structure
├── data/              # Dataset files
├── notebooks/         # Jupyter notebooks
├── src/               # Source code
├── models/            # Saved models
├── results/           # Evaluation outputs
└── README.md
How to Run
# Clone the repository
git clone https:

# Navigate to project folder


# Install dependencies
pip install -r requirements.txt

# Run the main script
python main.py
Key Highlights
Reduces labeling cost using Active Learning strategies
Improves performance with Transfer Learning
Efficient pipeline suitable for healthcare applications
Demonstrates real-world applicability under limited data conditions
Future Improvements
Improve F1-score using better class balancing techniques
Experiment with advanced architectures (e.g., transformers)
Deploy using Streamlit or Flask for real-time usage
Expand dataset for better generalization
