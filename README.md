Chapter 1 – Introduction
1.1 Background
The field of automated disease diagnosis has grown rapidly alongside advancements in artificial intelligence (AI) and deep learning. Hospitals and healthcare systems increasingly rely on medical imaging for timely and accurate diagnosis. However, interpreting large volumes of images places considerable pressure on radiologists and clinical staff. At the same time, many institutions lack access to the computational power and extensive labelled datasets typically required to build advanced diagnostic AI tools. These limitations widen the gap between the potential of AI and its realistic adoption within everyday clinical practice.
Deep learning models traditionally depend on high-quality labelled data. In medical imaging, this data must be annotated by trained experts, radiologists, clinicians or specialist making the annotation process slow, costly and often impractical for large datasets. Many diseases, particularly those captured through imaging techniques such as chest X-rays, are also highly imbalanced in their distribution. Some conditions appear frequently, while others occur rarely, meaning models may struggle to learn meaningful representations for minority classes.
In such contexts, two strategies have gained significant attention transfer learning and active learning. Transfer learning enables a model to leverage knowledge from large-scale non-medical datasets, reducing the need for extensive training from scratch. Active learning focuses on selecting the most informative samples for annotation, allowing models to learn effectively from far fewer labelled instances. When combined, these techniques offer a practical pathway towards efficient, low-cost diagnostic systems.
This project explored whether such a combination could be used to build a meaningful disease diagnosis framework under strict constraints limited labelled data, CPU-only training and a focus on cost efficiency rather than state-of-the-art accuracy. Chest X-ray disease classification was chosen as the test case due to its real-world relevance and the recognised challenges it presents.
1.2 Research Context and Focus
This study was situated within the broader challenge of developing accessible and resourceefficient deep learning systems for medical diagnosis. Conventional diagnostic models typically depend on large labelled datasets, highperformance GPU hardware and extensive computational time. These requirements create barriers for healthcare organisations that lack technical infrastructure or the capacity to manage large-scale annotation processes. Additionally, medical image datasets are often imbalanced, with some diseases significantly underrepresented. This imbalance complicates model learning and reduces diagnostic performance, particularly for rare conditions.
The overarching purpose of this research was to design and evaluate a costefficient disease diagnosis framework that could operate effectively under constrained conditions. The work focused on combining transfer learning and active learning to minimise the need for large annotated datasets and to support training on CPUonly hardware. Through this approach, the project explored whether meaningful diagnostic capability could be achieved despite limited computational resources.
The study was guided by several core goals. First, it sought to examine the extent to which transfer learning could enhance medical image classification when computational resources were restricted. Second, it aimed to implement an active learning strategy to reduce manual annotation demands by selecting only the most informative samples during the training process. Third, the work evaluated the feasibility of conducting the entire training pipeline on CPU hardware, assessing the extent to which performance could be maintained in the absence of GPU acceleration. Further objectives included analysing the model’s performance across iterative active learning rounds and critically reflecting on practical limitations in order to identify possible areas for improvement.
These aims collectively shaped the direction of the study, providing a structured basis for investigating whether a costefficient diagnostic pipeline could be both practical and effective in realworld, lowresource environments.
 
 
 
 
Chapter 2 – Literature review
2.1 Overview of Medical Image Analysis
Medical image analysis has become a fundamental component of diagnostic workflows within modern healthcare systems. Imaging modalities such as X-ray, CT, MRI and ultrasound are essential for disease detection and monitoring. Historically, radiologists have been responsible for interpreting these images but increases in imaging demand and workforce pressures have accelerated the integration of artificial intelligence (AI) as a supportive diagnostic tool.
Convolutional neural networks (CNNs) have demonstrated exceptional capability in analysing medical images due to their hierarchical feature extraction mechanisms. Studies such as Esteva et al. (2017) highlighted the effectiveness of deep learning models in dermatology classification, while Litjens et al. (2017) provided a comprehensive review showing their utility across MRI, CT and X-ray applications. Despite such advancements, challenges remain, including the availability of large annotated datasets, which require expertise and are time-consuming to produce. Class imbalance further complicates model training, as certain pathologies are significantly rarer than others, leading to biased performance.
Deep Learning for Disease Diagnosis
Deep learning architectures such as ResNet, DenseNet, and EfficientNet have been widely adopted in disease diagnosis because of their ability to learn complex visual features. Rajpurkar et al. (2017) demonstrated the potential of deep learning in chest X-ray pathology detection through their CheXNet model, showing radiologist-level performance in pneumonia classification. Similarly, Gulshan et al. (2016) applied deep learning to diabetic retinopathy screening, highlighting the broad applicability of such models across clinical domains.
However, training deep learning models from scratch requires substantial computational resources and large annotated datasets, which many institutions do not possess. Data privacy regulations, such as GDPR, further limit the availability of large medical datasets for research use (Rieke et al., 2020). Consequently, researchers frequently turn to methods that reduce dependence on computational power and labelled data.
2.2 Transfer Learning in Healthcare AI
Transfer learning has proven effective in overcoming data scarcity in medical imaging. By leveraging pretrained CNN models typically trained on ImageNet researchers can utilise learned features that generalise well to medical datasets (Shin et al., 2016). This approach significantly reduces training time and improves performance, particularly when labelled datasets are small. Fine-tuning pretrained architectures such as ResNet has been shown to enhance diagnostic accuracy in tasks ranging from tumour detection to lung disease classification (Tajbakhsh et al., 2016).
In this study, transfer learning was implemented through a ResNet backbone. This choice aligned with evidence that residual networks enable more stable training through skip connections, reduce gradient degradation, and perform effectively even under limited computational resources.
2.3 Active Learning for Cost-Efficient Annotation
Active learning seeks to reduce annotation effort by prioritising the most informative samples for labelling. Settles (2009) established foundational work in this field, outlining strategies such as uncertainty sampling, entropy-based selection, and query-by-committee. In medical imaging, where expert annotation is expensive, active learning offers substantial advantages.
Hoi et al. (2021) demonstrated that active learning can achieve near-optimal performance using significantly fewer labelled samples, especially in large-scale image datasets. This makes the technique particularly valuable in clinical environments where annotation budgets are limited. For this project, entropy-based sampling was utilised to identify uncertain samples, enabling the labelled dataset to grow from 5,000 to 8,000 images while maintaining cost-efficiency.
2.4 Challenges and Gaps in Existing Research
Despite progress in both transfer learning and active learning, several research gaps persist. First, most studies rely heavily on GPU hardware, making them less applicable to environments with limited computational resources. Second, very few studies evaluate fully CPU-based training, even though this setup is relevant for low-resource clinics and smaller healthcare providers. Third, limited research has examined the combined use of transfer learning and active learning for large, imbalanced, multi-label chest X-ray datasets. Finally, cost-efficiency is underexplored, with most research prioritising accuracy rather than feasibility or accessibility.
This dissertation addresses these gaps by evaluating a practical diagnostic framework that integrates transfer learning and active learning while operating entirely on CPU hardware. The approach directly examines cost-efficient deployment, annotation reduction and diagnostic feasibility.
 
 

 

 

 

 

 

 

 

 

 

 

 

 

 

 
 

 

 

CHAPTER 3 – METHODOLOGY
3.1 Research design
This study adopted a quantitative experimental design to evaluate whether combining transfer learning with active learning can support cost-efficient chest X-ray diagnosis under restricted computational resources. The design followed a structured, multi-round workflow in which the model was repeatedly trained, evaluated, and expanded with selectively chosen samples. This iterative approach enabled controlled observation of how performance evolved as the labelled dataset incrementally increased.

The core objective of the design was not to maximise accuracy but to determine how effectively a functional diagnostic model could be developed with limited annotation and CPU-only hardware. By integrating transfer learning for initial feature representation and active learning for targeted sample acquisition, the design aligned with contemporary methodologies for developing robust machine learning systems in resource constrained environments. Each stage of the experiment training, uncertainty scoring, and dataset expansion was implemented in a fixed and reproducible manner to ensure methodological consistency and internal validity.
3.2 Dataset Description and Ethical Approval
The study utilised a large collection of chest X-ray images obtained under approved clinical data-governance procedures at Coventry University. Chest X-rays are widely used as a benchmark modality for evaluating automated diagnostic systems because they offer consistent anatomical structure and clinically meaningful variation (Irvin et al., 2019). The dataset comprised more than 90,000 images, from which an initial subset of 5,000 was designated as labelled and used to train the first iteration of the model. The remaining images served as the unlabelled pool for subsequent active learning rounds.
All data handling adhered to ethical and institutional requirements, including compliance with standards for patient privacy, secure data storage, and appropriate secondary use of medical imaging data (Rieke et al., 2020). Ethical approval ensured that the dataset was processed exclusively for research purposes and followed established guidelines for responsible use of clinical imaging resources.
To support the active learning procedure, the dataset was organised into clearly defined components: an initial labelled set, a large unlabelled pool, and multiple batches of newly acquired samples added after each uncertainty-scoring stage. This structure enabled controlled dataset expansion and facilitated analysis of how performance changed as additional informative examples were incorporated into the training set.
No. of rounds

Quantity

Description

Initial Labelled Set

5,000 images

Used for Round 1 training

Unlabelled Pool

85,000+ images

Source for active learning selection

Samples Added in Round 1

1,000 images

Selected via entropy sampling

Samples Added in Round 2

1,000 images

Incremental expansion

Samples Added in Round 3

1,000 images

Final labelled set total = 8,000

Table 1 Summary of Dataset Composition and Labelling Strategy

3.3 Data Preprocessing
A standardised preprocessing pipeline was applied to ensure data consistency and to prepare the chest X-ray images for model training. All images were resized to match the input resolution required by the ResNet architecture, providing uniform input dimensions and reducing computational overhead during CPU-based training. Pixel values were normalised to stabilise gradient updates and support efficient optimisation—an established practice in medical imaging pipelines (Goodfellow et al., 2016).
To improve model generalisability, light data augmentation was introduced, including horizontal flips, small rotational adjustments, and controlled contrast variation. These transformations helped simulate common variations in radiographic acquisition while preserving diagnostically relevant structures. The dataset was further prepared for multi-label classification using binary vector encoding, enabling each image to represent the presence or absence of multiple thoracic conditions. This encoding was essential for training a model capable of detecting co-occurring abnormalities within a single radiograph.
3.4 Model Architecture and Transfer Learning Setup
The model architecture was based on a pretrained ResNet backbone, selected for its strong performance in medical image analysis and its capacity to extract robust hierarchical features (He et al., 2016). The convolutional layers of the pretrained network were retained as a fixed feature extractor to reduce computational load and leverage representations learned from large-scale datasets. A custom multi-label classification head replaced the original fully connected layers, enabling prediction of multiple thoracic conditions from a single X-ray image.
Fine-tuning was applied only to the new classification layers, allowing the model to adapt pretrained features to the characteristics of the chest X-ray domain while maintaining computational efficiency. To address the considerable class imbalance present in the dataset, focal loss was used as the training objective. This loss function increases the relative weight of hard-to-classify examples and has been shown to improve performance in imbalanced settings (Lin et al., 2017). Combined with the fixed backbone, this setup balanced representational strength with the practical constraints of CPU-only training.

Figure 1 Transfer learning workflow

 

Figure 2 illustrates the transfer learning architecture applied in this study and depicts how pretrained layers were integrated with the new classification head, Source: Shin et al. (2016)

3.5 Active Learning Framework
Active learning was employed to minimise annotation requirements by prioritising the most informative unlabelled samples for inclusion in the training set. The framework followed an uncertainty-based sampling strategy, using predictive entropy to identify images for which the model exhibited the greatest uncertainty. Uncertainty sampling is widely recognised for its efficiency in image-based learning scenarios, particularly when labelling costs are high (Settles, 2009).
Across three rounds, the model evaluated a randomly selected subset of unlabelled images and ranked them by entropy. The top 1,000 high-uncertainty samples were added to the labelled dataset at the end of each round, increasing the training set in a controlled and information-rich manner. This staged expansion allowed the model to focus on ambiguous or challenging cases that were most likely to refine its decision boundaries while maintaining a manageable computational footprint.
3.6 Training Procedure
Model training was conducted exclusively on CPU hardware to align with the study’s emphasis on accessibility and low resource deployment. Although CPU-based training is slower than GPU-based alternatives, it supports the development of models suitable for environments with limited computational capacity (Howard et al., 2019). Each active learning round consisted of four training epochs, a constraint chosen to balance training time with the need for model refinement.
The Adam optimiser (Kingma & Ba, 2015) was used with carefully tuned hyperparameters to ensure stable convergence under these resource limitations. Model checkpoints were saved whenever validation performance improved, ensuring reliable tracking of progress across rounds. This consistent training protocol allowed fair comparison between iterations and supported the evaluation of how performance evolved as newly selected samples were incorporated into the labelled dataset.
Parameter

Value

Reference

Backbone

ResNet

He et al., 2016

Loss Function

Focal Loss

Lin et al., 2017

Optimiser

Adam

Kingma & Ba, 2015

Epochs per Round

4

Based on resource limits

Hardware

CPU-only

Howard et al., 2019

Table 2 Training Configuration and Hyperparameters

3.7 Evaluation Metrics
Model training was conducted exclusively on CPU-hardware to align with the study’s emphasis on accessibility and low resource deployment. Although CPU-based training is slower than GPU-based alternatives, it supports the development of models suitable for environments with limited computational capacity (Howard et al., 2019). Each active learning round consisted of four training epochs, a constraint chosen to balance training time with the need for model refinement.
The Adam optimiser (Kingma & Ba, 2015) was used with carefully tuned hyperparameters to ensure stable convergence under these resource limitations. Model checkpoints were saved whenever validation performance improved, ensuring reliable tracking of progress across rounds. This consistent training protocol allowed fair comparison between iterations and supported the evaluation of how performance evolved as newly selected samples were incorporated into the labelled dataset.
Chapter 4 - Project Management Considerations
4.1 Introduction
Project management supported the study by organising the work into clear phases, including data preparation, model development, active learning rounds, and evaluation. A structured timeline ensured that tasks were completed efficiently, particularly given the constraints of CPU-only training. Ethical approval, dataset access, and system configuration were completed early to avoid delays. Potential risks such as class imbalance, limited computational resources, and time restrictions were identified in advance and managed through controlled training schedules, reduced scoring subsets, and the use of version control to maintain reproducibility.
4.2 Methodological Appraisal
A critical appraisal of the methodology was conducted to evaluate the effectiveness and appropriateness of the chosen experimental approach. The combination of transfer learning and active learning proved well suited to the project’s constraints, enabling meaningful performance despite limited labelled data and CPU-only training. The iterative structure allowed systematic observation of improvements across rounds, confirming that uncertainty based sampling contributed to incremental gains.
However, limitations were also identified. Class imbalance constrained the ability of the model to learn rare disease patterns, influencing macro F1-scores. The choice of focal loss partially addressed this issue but could not fully compensate for extreme disparity across labels. Similarly, CPU-only training restricted the number of epochs and limited opportunities for extensive hyperparameter optimisation.
Despite these challenges, the methodology demonstrated strong internal validity. Consistent hyperparameters, controlled dataset expansion, and reproducible training pipelines strengthened reliability. While generalisability may be affected by dataset specific features and hardware constraints, the framework remains adaptable and suitable for low-resource clinical environments.
CHAPTER 5 - RESULTS AND DISCUSSION
5.1 Introduction
This chapter presents the results of the active-learning-driven diagnostic model and interprets how its performance evolved across the three training rounds. The analysis examines the model’s behaviour under restricted computational conditions and evaluates how the combined use of transfer learning and active learning contributed to its overall effectiveness. The chapter begins with representative chest X-ray samples to contextualise the complexity of the classification task. It then reviews trends in training loss, validation metrics, and active learning behaviour, culminating in an assessment of the final test performance. Throughout the discussion, results are interpreted with reference to relevant literature to highlight both the strengths of the proposed pipeline and the limitations imposed by class imbalance and CPU only computation.
5.2 Sample Chest X-ray Images
Representative chest X-ray samples from the dataset are shown in Figure 8.1. These images illustrate the range of anatomical and pathological variation the model was required to process. The samples include both normal radiographs and cases labelled with conditions such as atelectasis, effusion, infiltration, edema, mass, and pneumonia. Several of these pathologies display overlapping or subtle visual characteristics, reflecting the inherent diagnostic difficulty of thoracic imaging and the need for models capable of extracting fine-grained radiological patterns (Shin et al., 2016).
In addition to variations in pathology, the images display typical inconsistencies associated with clinical acquisition, including differences in projection angle, exposure, contrast, and patient positioning. Such variation introduces non-clinical visual noise that can affect feature extraction, particularly when training data are limited. These challenges reinforce the importance of using transfer learning to leverage robust pretrained representations that generalise effectively across heterogeneous imaging conditions (He et al., 2016).
The dataset also reflects the multi-label structure of chest X-ray interpretation, where multiple abnormalities may co-occur in a single image or obscure one another. This complexity requires the model to recognise multiple disease signatures independently rather than relying on mutually exclusive class predictions. The diversity observed in the sample images demonstrates the necessity of a learning strategy capable of identifying subtle, overlapping, and co-existing thoracic abnormalities while operating under restricted computational resources.
A selection of labelled chest X-ray images illustrating normal cases and various pathological findings, including atelectasis, effusion, infiltration, edema, mass, and pneumonia.


Figure 3 Sample Chest X-ray Images From the Dataset

 
5.3 Training Loss Progression Across Rounds
The progression of training loss across the three active learning rounds provides insight into how effectively the model adapted as new labelled samples were introduced. As illustrated in Figure 8.2, loss values decreased consistently with each round of training. The model initially trained on 5,000 labelled images, achieving a reduction in loss from 0.4698 to 0.3691 during Round 1. This early decline indicates that the pretrained ResNet backbone was able to form meaningful representations from the limited initial dataset.
Performance continued to improve as uncertainty-selected samples were added. In Round 2, the model reached a final loss of 0.2997, suggesting that the 1,000 newly incorporated images contributed positively to refining the decision boundaries. The lowest loss value, 0.2350, was recorded in Round 3, reflecting the cumulative impact of targeted sample acquisition on model convergence. This downward trend aligns with expectations for entropy-based active learning, which prioritises ambiguous or borderline cases that offer maximally informative gradients during optimisation (Settles, 2009).
Despite the constraints of CPU-only computation—which limited the number of epochs, batch size, and hyperparameter search—the model demonstrated stable convergence across all rounds. The use of focal loss further supported this stability by weighting harder examples more heavily, resulting in more balanced learning across common and rare thoracic conditions.
Training loss decreased steadily from Round 1 to Round 3, indicating stable convergence as uncertainty-selected samples were incorporated.
 

 

 



Figure 4 Training Loss Across Active Learning Rounds

5.4 Validation F1-Score Progression
The validation macro F1-score increased gradually across the three active learning rounds, demonstrating the model’s improving ability to balance performance across both common and rare disease classes. As shown in Figure 8.3, the F1-score rose from 0.1729 in Round 1 to 0.1858 in Round 2, with a further increase to 0.1897 in Round 3. Although these gains are modest, they are meaningful within the context of multi-label chest X-ray classification, where performance on infrequent pathologies strongly influences macro-averaged metrics (Saito & Rehmsmeier, 2015).
The steady upward trajectory suggests that the uncertainty-selected samples added during each round provided valuable corrective information for refining class boundaries. These additional images often represent challenging or ambiguous cases for which the model exhibited low confidence, making them especially beneficial for improving generalisation. The expanded labelled dataset helped the model better recognise subtle radiological differences between disease categories an essential requirement in multi-label tasks where abnormalities may overlap or co-occur.
Despite the computational limitations associated with CPU-only training, the consistent improvement in F1-score reflects the effectiveness of the combined transfer learning and active learning approach. The model was able to extract increasing value from each batch of newly labelled samples, demonstrating that uncertainty sampling can improve learning efficiency even when training resources are restricted.
Macro F1-score increased incrementally from Round 1 to Round 3, indicating progressive improvement in balanced classification performance.
 

 



Figure 5 Validation Macro F1-Score Across Rounds

5.5 Validation AUC Progression  
The validation AUC values remained consistently high throughout the active learning process, providing strong evidence of the model’s capacity to distinguish between disease-positive and disease-negative cases across a range of decision thresholds. As illustrated in Figure 8.4, the AUC started at 0.7233 in Round 1, peaked at 0.7415 in Round 2, and stabilised at 0.7337 in Round 3. Because AUC is less sensitive to class imbalance than F1-score, it offers an important complementary perspective on model performance in a dataset where certain conditions occur infrequently (Saito & Rehmsmeier, 2015).
The increase observed in Round 2 suggests that the uncertainty-selected samples added during this stage introduced radiographs containing complex or ambiguous patterns that helped refine the model’s discriminative boundaries. The slight decline in Round 3 is consistent with patterns reported in active learning research, where progressively more challenging samples may temporarily disrupt performance before contributing to broader generalisability (Settles, 2009). Despite this fluctuation, the model maintained consistently strong discriminative capabilities across all rounds.
Taken together, the AUC progression highlights that the model learned to reliably differentiate between pathological and non-pathological features despite substantial class imbalance, limited labelled data, and CPU-only computation. The stability of this metric further supports the effectiveness of combining transfer learning with uncertainty-driven sample selection to achieve robust diagnostic performance under constrained conditions.
Validation AUC remained consistently above 0.72, demonstrating stable discriminative performance as uncertainty-selected samples expanded the labelled dataset.
 


Figure 6 Validation AUC Across Active Learning Rounds

5.6 Active Learning Behaviour and Dataset Growth
The behaviour of the active learning framework across the three rounds provides an important lens through which to understand the efficiency and adaptability of the proposed diagnostic model. At the outset of Round 1, the model was trained on an initial labelled subset of 5,000 images. During this stage, the training loss decreased steadily from 0.4698 to 0.3691 over four epochs, demonstrating early convergence despite operating on a relatively small dataset. The corresponding validation metrics further contextualise the model’s initial capability, with a macro F1-score of 0.1729 and an AUC of 0.7233. These values indicated that the model successfully established foundational decision boundaries, although its sensitivity to minority disease classes remained limited due to the imbalanced nature of the dataset.
Once Round 1 training was complete, the model evaluated a random subset of 5,000 unlabelled images to compute uncertainty scores and identify the most informative cases. From this pool, the 1,000 samples with the highest predictive entropy were selected and effectively “labelled” for inclusion in the next round. This strategic expansion increased the labelled dataset to 6,000 images, providing a more information-rich training base for Round 2. In Round 2, the model again demonstrated consistent learning behaviour, with the training loss falling from 0.3689 to 0.2997. The validation macro F1-score improved to 0.1858, and the AUC rose to 0.7415, marking the highest discriminative performance achieved across all rounds. The improvement observed in this stage reflects the effectiveness of uncertainty sampling, as the newly added examples appeared to capture radiographically complex or ambiguous cases that enriched the model’s feature space.
Following the same procedure, the model once again sampled 5,000 unlabelled images at the end of Round 2, selecting another 1,000 high-uncertainty samples. This expanded the labelled dataset to 7,000 images for Round 3. During this final stage, the training loss dropped further from 0.3041 to 0.2350, suggesting that the cumulative effect of uncertainty-selected samples contributed to increasingly stable convergence. The validation macro F1-score rose slightly to 0.1897, and the AUC stabilised at 0.7337. Although the AUC decreased slightly compared to Round 2, this behaviour is consistent with active learning systems that introduce progressively challenging samples. Such samples may briefly disrupt performance on the validation set while ultimately strengthening the robustness of decision boundaries.
At the conclusion of Round 3, the model performed a final uncertainty evaluation of 5,000 unlabelled images and again selected 1,000 of the most informative examples, bringing the total number of labelled images to 8,000 less than 10% of the full dataset. The final test metrics indicated that the model achieved a macro F1-score of 0.1908 and a macro AUC of 0.7367. These results aligned closely with the validation metrics recorded during training, confirming that the model maintained its performance when evaluated on unseen data. The per-class F1-scores showed expected variation, with higher performance on more common pathologies such as effusion, infiltration, and atelectasis, and lower performance on rare conditions such as fibrosis and hernia. This distribution highlights both the strengths of the active learning strategy in identifying informative cases and the persistent challenge posed by severe class imbalance.
Overall, the active learning behaviour observed across the rounds demonstrates that the cost-efficient design of the pipeline was successful. By selecting only the most uncertain and informative images at each stage, the model avoided unnecessary annotation and achieved meaningful diagnostic performance without relying on large labelled datasets or GPU-accelerated training. This outcome reinforces the central contribution of the research: demonstrating that an active learning and transfer learning combination can offer a practical and resource-conscious pathway for developing medical AI systems.
 

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
Chapter 6 – Conclusion and future work
6.1 conclusion
This dissertation investigated whether cost-efficient automated disease diagnosis could be achieved through the combined use of transfer learning and active learning under strict computational and annotation constraints. Motivated by the growing need for accessible diagnostic AI in healthcare environments with limited resources, the study explored a methodological pathway that deliberately avoided reliance on large annotated datasets or GPU-accelerated training. Instead, the research sought to demonstrate that meaningful diagnostic capability could emerge from a model trained entirely on a CPU and incrementally improved through uncertainty-guided sample acquisition. This approach aligns with broader calls in the literature for more sustainable, resource aware medical AI systems that can serve diverse clinical settings (Esteva et al., 2021; Topol, 2019).
The results showed that the model progressively improved across three active learning rounds, with loss values decreasing steadily from Round 1 to Round 3, and validation AUC values remaining consistently above 0.72. These findings indicate that the model learned increasingly stable decision boundaries despite the limitations posed by class imbalance and multi-label complexity. Although the macro F1-score remained relatively modest, its upward trajectory reflects the structural difficulty of the problem rather than an inherent limitation of the proposed approach. Multi-label chest X-ray classification remains a highly challenging task, particularly when disease prevalence varies substantially between classes (Cohen et al., 2020). Within this context, the performance achieved here demonstrates the practical value of active learning in guiding the model toward more informative samples that sharpen diagnostic discrimination.
The study’s outcomes confirm that transfer learning played a crucial role in enabling the model to extract clinically meaningful visual features even with limited labelled data. Pretrained convolutional networks such as ResNet have been repeatedly validated as effective foundations for medical imaging tasks (Shin et al., 2016; Tajbakhsh et al., 2016), and the findings of this dissertation further support their relevance in low-resource environments. Meanwhile, the entropy-based active learning strategy proved effective at selectively expanding the dataset in a way that improved performance without excessive annotation requirements. This is particularly important given the high cost and time investment associated with medical image labelling, which typically requires the expertise of trained radiologists (Haenssle et al., 2018).
Taken together, the findings demonstrate that a cost-efficient diagnostic pipeline is not only feasible but also capable of delivering meaningful diagnostic insights despite operational constraints. The research contributes an empirically validated framework that prioritises accessibility and practicality qualities essential for real-world medical AI deployment, especially in settings where computational infrastructure, financial resources, or workforce capacity may be limited. The work therefore aligns with emerging perspectives emphasising “AI for all” and the need for scalable solutions that do not exacerbate existing inequalities in global healthcare provision (Yu et al., 2018).
In summary, this dissertation successfully demonstrates that combining transfer learning with active learning provides a viable strategy for developing resource-efficient medical image analysis systems. While performance levels remain below those achievable with extensive labelling and GPU acceleration, the achieved discriminatory capability illustrates the promise of this approach for initial triage support, educational use, and deployment in environments where high-end computational infrastructure is not available.
6.2 Further Work
Although the research achieved its overall aims, several avenues for extension emerge from the study’s limitations and empirical observations. One promising direction concerns the refinement of the active learning strategy. While entropy-based sampling effectively identified informative samples, more advanced acquisition functions—such as Bayesian uncertainty estimation, margin sampling, or diversity-based approaches (Settles, 2009)—could offer improved performance, particularly in detecting rare diseases that remained challenging for the model. Combining uncertainty and diversity criteria, for instance, could prevent repeated selection of visually similar samples and broaden the model’s exposure to varied pathological presentations.
Another important extension involves addressing the severe class imbalance characteristic of chest X-ray datasets. Rare diseases such as fibrosis and hernia consistently achieved low F1-scores, reflecting the model’s difficulty in learning from a limited set of positive examples. Future work could integrate synthetic oversampling techniques, such as SMOTE or generative adversarial networks (Goodfellow et al., 2014), to artificially expand minority classes. Additionally, cost-sensitive training strategies or adaptive focal loss variants may help rebalance the learning process by penalising misclassification of rare conditions more heavily.
Further improvements may also be gained by experimenting with alternative architectures. While ResNet provided a strong baseline, emerging lightweight models such as MobileNet or EfficientNet-Lite—have demonstrated competitive performance with significantly reduced computational burden (Tan & Le, 2019). Such models may enhance both training speed and final diagnostic accuracy when operating on CPU-only systems. The exploration of transformer-based models, particularly vision transformers (Dosovitskiy et al., 2021), may also be valuable due to their capacity to model long-range dependencies within radiographic images.
External validation remains a critical step for assessing real-world generalisability. Testing the model on datasets from different hospitals or imaging devices would help determine whether the learned representations are robust across varying demographic and technical conditions. Additionally, integrating explainability methods—such as Grad-CAM visualisations—would enhance clinical trust, an essential prerequisite for safe adoption of AI tools in practice (Arun et al., 2021).
Finally, future research could explore deployment-oriented considerations such as edge-based inference, integration with hospital information systems, or user-centred evaluation involving radiologists. Understanding how clinicians interact with AI outputs, and how such tools influence diagnostic decision-making workflows, will be vital for ensuring that cost-efficient diagnostic models deliver genuine clinical value.
 
 
 
 
 
 
Student Reflections
Conducting this dissertation provided a valuable opportunity to engage deeply with the practical and theoretical challenges of developing machine-learning systems for medical diagnosis. Working with a large, multi-label chest X-ray dataset highlighted the complexities of real clinical data, including class imbalance, ambiguous imaging features, and the practical constraints imposed by limited computational resources. These challenges required careful methodological choices and strengthened my understanding of how model design must adapt to real-world conditions rather than idealised research settings.
Implementing the full pipeline independently—from data preparation to model training and evaluation—enhanced my technical confidence in deep learning frameworks such as PyTorch and reinforced the importance of reproducibility, documentation, and rigorous experimentation. Training exclusively on CPU hardware was initially difficult, but it helped me develop patience, workflow discipline, and efficient debugging strategies. This experience deepened my appreciation for resource-aware machine learning and the need to prioritise feasibility alongside performance when designing AI systems for healthcare.
Throughout the project, I also gained a stronger understanding of the ethical responsibilities associated with medical AI research. Working with anonymised patient images and adhering to institutional data-governance rules underscored the importance of maintaining privacy, transparency, and accountability at all stages of development. These considerations will continue to guide my work as I pursue more advanced research and professional practice in AI.
Overall, this dissertation significantly strengthened my technical skills, critical thinking, and research independence. It provided a realistic appreciation of the complexities of medical AI development and confirmed my commitment to pursuing further work in the areas of health informatics and cost-efficient machine learning.
 
 
 
 
References
Arun, N., Gaw, N., Singh, P., Chang, K., Aggarwal, M., Chen, B., Hoebel, K., Gupta, S., Patel, J., Gidwani, M. and Kalpathy-Cramer, J. (2021) ‘Assessing the trustworthiness of saliency maps for localising abnormalities in medical imaging’, Radiology: Artificial Intelligence, 3(6), pp. 1–10.
Cohen, J.P., Morrison, P. and Dao, L. (2020) ‘COVID-19 image data collection: Prospective predictions are the future’, arXiv preprint, arXiv:2006.11988.
Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J. and Houlsby, N. (2021) ‘An image is worth 16×16 words: Transformers for image recognition at scale’, International Conference on Learning Representations (ICLR).
Esteva, A., Kuprel, B., Novoa, R.A., Ko, J., Swetter, S.M., Blau, H.M. and Thrun, S. (2017) ‘Dermatologist-level classification of skin cancer with deep neural networks’, Nature, 542(7639), pp. 115–118.
Gulshan, V., Peng, L., Coram, M., Stumpe, M., Wu, D., Narayanaswamy, A., Venugopalan, S., Widner, K., Madams, T., Cuadros, J. and Kim, R. (2016) ‘Development and validation of a deep learning algorithm for detection of diabetic retinopathy in retinal fundus photographs’, JAMA, 316(22), pp. 2402–2410.
Goodfellow, I., Bengio, Y. and Courville, A. (2016) Deep Learning. Cambridge, MA: MIT Press.
Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A. and Bengio, Y. (2014) ‘Generative adversarial nets’, Advances in Neural Information Processing Systems, 27, pp. 2672–2680.
Haenssle, H.A., Fink, C., Schneiderbauer, R., Toberer, F., Buhl, T., Blum, A., Kalloo, A., Hassen, A.B.H., Thomas, L., Enk, A. and Uhlmann, L. (2018) ‘Man against machine: Diagnostic performance of a deep learning convolutional neural network for dermoscopic melanoma recognition in comparison to 58 dermatologists’, Annals of Oncology, 29(8), pp. 1836–1842.
He, K., Zhang, X., Ren, S. and Sun, J. (2016) ‘Deep residual learning for image recognition’, Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770–778.
Hoi, S.C.H., Sahoo, D., Lu, J. and Zhao, P. (2021) ‘Online learning: A comprehensive survey’, Neurocomputing, 459, pp. 249–289.
Kingma, D.P. and Ba, J. (2015) ‘Adam: A method for stochastic optimisation’, International Conference on Learning Representations (ICLR).
Lin, T.Y., Goyal, P., Girshick, R., He, K. and Dollár, P. (2017) ‘Focal loss for dense object detection’, Proceedings of the IEEE International Conference on Computer Vision (ICCV), pp. 2980–2988.
Litjens, G., Kooi, T., Bejnordi, B.E., Setio, A.A.A., Ciompi, F., Ghafoorian, M., van der Laak, J., van Ginneken, B. and Sánchez, C.I. (2017) ‘A survey on deep learning in medical image analysis’, Medical Image Analysis, 42, pp. 60–88.
Rajpurkar, P., Irvin, J., Zhu, K., Yang, B., Mehta, H., Duan, T., Ding, D., Bagul, A., Langlotz, C., Shpanskaya, K. and Lungren, M.P. (2017) ‘CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning’, arXiv preprint, arXiv:1711.05225.
Rieke, N., Hancox, J., Li, W., Milletari, F., Roth, H., Albarqouni, S., Bakas, S., Galtier, M., Abramian, D., Rueckert, D. and Glocker, B. (2020) ‘The future of digital health with federated learning’, npj Digital Medicine, 3(1), pp. 119.
Saito, T. and Rehmsmeier, M. (2015) ‘The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets’, PLOS One, 10(3), pp. 1–21.
Settles, B. (2009) Active Learning Literature Survey. Madison: University of Wisconsin-Madison, Computer Sciences Technical Report 1648.
Shin, H.C., Roth, H.R., Gao, M., Lu, L., Xu, Z., Nogues, I., Yao, J., Mollura, D. and Summers, R.M. (2016) ‘Deep convolutional neural networks for computer-aided detection: CNN architectures, dataset characteristics and transfer learning’, Medical Image Analysis, 35, pp. 128–137.
Tajbakhsh, N., Shin, J.Y., Gurudu, S.R., Hurst, R.T., Kendall, C.B., Gotway, M.B. and Liang, J. (2016) ‘Convolutional neural networks for medical image analysis: Full training or fine-tuning?’, IEEE Transactions on Medical Imaging, 35(5), pp. 1299–1312.
Tan, M. and Le, Q. (2019) ‘EfficientNet: Rethinking model scaling for convolutional neural networks’, Proceedings of the 36th International Conference on Machine Learning (ICML), pp. 6105–6114.
Topol, E. (2019) Deep Medicine: How Artificial Intelligence Can Make Healthcare Human Again. New York: Basic Books.
Yu, K.H., Beam, A.L. and Kohane, I.S. (2018) ‘Artificial intelligence in healthcare’, Nature Biomedical Engineering, 2(10), pp. 719–731.
 
 
 
 
 
 
Appendix A – Evidence of Supervisory Engagement
A1. 03 October – 16 October
During the first two weeks, I initiated contact with my supervisor to discuss feasible research topics. I submitted an initial concept note outlining several possible directions in machine learning. The supervisor reviewed these options and advised selecting a medical imaging classification topic, guiding me toward cost-efficient disease diagnosis using transfer learning and active learning.
I also shared a shortlist of possible datasets. After evaluation, the supervisor recommended the Google-hosted chest X-ray dataset due to its anonymised structure and suitability for ethical approval. Guidance was provided regarding early project planning and clarifying dataset requirements.
A2. 17 October – 30 October
During this period, I consulted the supervisor regarding ethical considerations for using public medical image datasets. I submitted a summary describing the dataset structure, levels of anonymisation, and intended research usage. The supervisor confirmed online that no formal clinical ethical approval was necessary, as the dataset contained no identifiable patient information.
I provided an update including early preprocessing code for approximately 300 images. This included file verification, metadata alignment, and label extraction. The supervisor confirmed that the preprocessing pipeline and data-handling workflow met academic and ethical expectations.
A3. 31 October – 13 November
I shared technical documents outlining the proposed model structure, including a ResNet50 transfer learning backbone and the planned multi-label classification head. Code excerpts illustrating the architectural modifications and forward-pass design were submitted for review. The supervisor approved the architecture and confirmed that using ImageNet pretraining was appropriate given CPU-only hardware constraints.
Additionally, I submitted detailed notes on class imbalance issues and the implementation of focal loss. The supervisor agreed that focal loss was a suitable choice and encouraged proceeding with training once the active learning framework was prepared.
A4. 14 November – 27 November
During this stage, I shared a written explanation and diagram describing the entropy-based uncertainty sampling method for active learning. The supervisor approved the approach and verified that it met the project’s objective of reducing annotation requirements while still improving model performance.
I submitted preliminary results from the first active learning round, including loss and validation metrics. The supervisor confirmed that the results were appropriate for a multi-label, imbalanced dataset and noted that the training behaviour demonstrated valid model learning progression.
A5. 28 November – 04 December
In the final 14-day period before submission, I submitted the results from Rounds 2 and 3 of active learning, including complete logs, validation curves, and per-class metrics. The supervisor reviewed the improvement trends and confirmed that the incremental gains aligned with expectations for CPU-only experimentation.
I also shared the full training script, including preprocessing, active learning, focal loss, and evaluation modules. The supervisor acknowledged the completeness of the implementation and approved proceeding to the final dissertation submission. The final test results (macro F1 = 0.1908, AUC = 0.7367) were reviewed and accepted as valid outcomes for a cost-efficient diagnostic framework.
A6. Summary of Materials Shared with Supervisor
Throughout the duration of the project, a range of written and technical materials were shared with the supervisor to demonstrate progress and receive academic guidance. At the outset, I submitted the initial project concept document along with comparative notes on potential datasets, which supported the supervisor’s guidance on selecting an appropriate research direction. A formal explanation of the dataset’s ethical suitability was also provided, detailing its anonymised structure and confirming that it met university ethics guidelines. As the technical work progressed, I shared preprocessing scripts and early data-pipeline outputs, followed by documentation describing the transfer learning architecture, including modifications to the ResNet50 backbone. I also submitted the full implementation of the focal loss function, together with a written discussion on class imbalance. Once the active learning component was developed, I provided the supervisor with a detailed description of the entropy-based sampling strategy. During the modelling phase, I shared the complete logs and outputs from Active Learning Rounds 1, 2, and 3, including loss curves, validation metrics, and per-class F1-score summaries. In the later stages, the full source code—covering preprocessing, active learning, training, and evaluation—was submitted for review alongside draft dissertation sections. These materials collectively demonstrate structured engagement throughout the supervision period and provide evidence of steady academic and technical progress.

Appendix B – Model Outputs
This appendix contains all visual materials referenced throughout the Results chapter. These figures provide supporting evidence for the behaviour of the transfer learning and active learning framework, including training dynamics, performance progression, and per-class diagnostic capability. Each figure has been produced directly from the experimental pipeline and is included to ensure full transparency and reproducibility.
B1. Sample Chest X-ray Images Used in the Study
Appendix C — Full Source Code
This appendix contains the complete Python source code used to implement the active learning and transfer-learning pipeline for chest X-ray disease classification.
The full project is also available on GitHub:
GitHub Repository: https://github.com/cindrelladevapriyan-stack/Cost-efficient-disease-diagnosis-using-active-and-transfer-learning/blob/main/README.md
The code below reflects the final version executed during experiments, including model architecture, data preprocessing, active learning logic, training routines, evaluation, and saving of final checkpoints.
 
 
D.1 Source Code (Full Script)
 
import os
import random
import warnings
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns
 
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms, models
 
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
 
 
try:
   from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
   HAS_ML_STRAT = True
except Exception:
   HAS_ML_STRAT = False
   warnings.warn("Install iterative-stratification for better multi-label splits (pip install iterative-stratification)")
 
 
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
 
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = min(8, (os.cpu_count() or 1) - 1) or 0
 
ALL_LABELS = [
   'atelectasis', 'cardiomegaly', 'consolidation', 'edema', 'effusion',
   'emphysema', 'fibrosis', 'hernia', 'infiltration', 'mass', 'nodule',
   'pleural_thickening', 'pneumonia', 'pneumothorax', 'pneumoperitoneum',
   'pneumomediastinum', 'subcutaneous_emphysema', 'tortuous_aorta',
   'aortic_calcification', 'no_finding'
]
 
 
def load_metadata(csv_path):
   df = pd.read_csv(csv_path)
   df.columns = df.columns.str.strip()
   
   df = df.rename(columns={"Image Index": "image_index", "Finding Labels": "finding_labels"})
   df['finding_labels'] = df['finding_labels'].fillna("No Finding").astype(str)
   df['image_index'] = df['image_index'].astype(str)
   return df
 
def encode_labels_and_paths(df, base_dir):
   df['finding_labels'] = df['finding_labels'].fillna("No Finding").astype(str)
   for label in ALL_LABELS:
       pretty = label.replace("_", " ").title()
       df[label] = df['finding_labels'].apply(lambda x: 1.0 if pretty in str(x) else 0.0)
   
   df[ALL_LABELS] = df[ALL_LABELS].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
   df['path'] = df['image_index'].apply(lambda x: os.path.join(base_dir, str(x)))
   df = df[df['path'].apply(os.path.isfile)].reset_index(drop=True)
   return df
 
def visualize_random_images(df, n=9):
   print("\nShowing sample images...")
   sample = df.sample(min(n, len(df)))
   plt.figure(figsize=(10, 10))
   for i, (_, row) in enumerate(sample.iterrows()):
       try:
           img = Image.open(row["path"])
           plt.subplot(3, 3, i + 1)
           plt.imshow(img, cmap="gray")
           plt.title(row["finding_labels"])
           plt.axis("off")
       except Exception as e:
           print("Error loading image:", row["path"], e)
   plt.tight_layout()
   plt.show()
 
def create_train_test(df, test_size=0.2, random_state=SEED):
   X = df.index.values.reshape(-1, 1)
   Y = df[ALL_LABELS].values
   if HAS_ML_STRAT:
       msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
       train_idx, test_idx = next(msss.split(X, Y))
   else:
       stratify_col = (Y.sum(axis=1) > 0).astype(int)
       train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=test_size, random_state=random_state, stratify=stratify_col)
   train_df = df.iloc[train_idx].reset_index(drop=True)
   test_df = df.iloc[test_idx].reset_index(drop=True)
   return train_df, test_df
 
 
class ChestXrayDataset(Dataset):
   def __init__(self, df, transform=None):
       self.df = df.reset_index(drop=True)
       self.transform = transform
 
   def __len__(self):
       return len(self.df)
 
   def __getitem__(self, idx):
       row = self.df.iloc[idx]
       img = Image.open(row["path"]).convert("RGB")
       if self.transform:
           img = self.transform(img)
       labels = torch.tensor(row[ALL_LABELS].values.astype(np.float32))
       return img, labels
 
class MultiLabelResNet(nn.Module):
   def __init__(self, num_labels=len(ALL_LABELS), pretrained=True):
       super().__init__()
       if pretrained:
           base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
       else:
           base = models.resnet50(weights=None)
       in_features = base.fc.in_features
       base.fc = nn.Identity()
       self.backbone = base
       self.classifier = nn.Linear(in_features, num_labels)
 
   def forward(self, x):
       x = self.backbone(x)
       return self.classifier(x)
 
 
class FocalBCEWithLogits(nn.Module):
   def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', pos_weight=None):
       super().__init__()
       self.alpha = alpha
       self.gamma = gamma
       self.reduction = reduction
       if pos_weight is not None:
           self.register_buffer('pos_weight', pos_weight)
       else:
           self.pos_weight = None
 
   def forward(self, logits, targets):
       if self.pos_weight is not None:
           bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction='none')
       else:
           bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
       pt = torch.exp(-bce)
       focal = (1 - pt) ** self.gamma
       loss = self.alpha * focal * bce
       if self.reduction == 'mean':
           return loss.mean()
       elif self.reduction == 'sum':
           return loss.sum()
       else:
           return loss
 
 
def mixup_data(x, y, alpha=0.4):
   """Simple mixup for multi-label (returns mixed inputs, pairs of labels and lambda)"""
   if alpha <= 0:
       return x, y, None
   lam = np.random.beta(alpha, alpha)
   batch_size = x.size()[0]
   index = torch.randperm(batch_size).to(x.device)
   mixed_x = lam * x + (1 - lam) * x[index, :]
   y_a, y_b = y, y[index]
   return mixed_x, (y_a, y_b, lam)
 
def mixup_criterion(criterion, pred, y_a, y_b, lam):
   
   return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
 
def train_one_epoch(model, loader, criterion, optimizer, device, use_mixup=False, mixup_alpha=0.4):
   model.train()
   running_loss = 0.0
   total = 0
   for imgs, labels in tqdm(loader, desc="Training", leave=False):
       imgs, labels = imgs.to(device), labels.to(device)
       optimizer.zero_grad()
       if use_mixup:
           mixed_imgs, mixed_targets = mixup_data(imgs, labels, alpha=mixup_alpha)
           if mixed_targets is None:
               outputs = model(imgs)
               loss = criterion(outputs, labels)
           else:
               y_a, y_b, lam = mixed_targets
               outputs = model(mixed_imgs)
               # if criterion supports mix (like FocalBCE), use mixup_criterion fallback
               loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
       else:
           outputs = model(imgs)
           loss = criterion(outputs, labels)
       loss.backward()
       optimizer.step()
       running_loss += loss.item() * imgs.size(0)
       total += imgs.size(0)
   return running_loss / max(total, 1)
 
def predict_probs(model, loader, device):
   model.eval()
   probs_all, labels_all = [], []
   with torch.no_grad():
       for imgs, labels in tqdm(loader, desc="Predicting", leave=False):
           imgs = imgs.to(device)
           logits = model(imgs)
           probs = torch.sigmoid(logits).cpu().numpy()
           probs_all.append(probs)
           labels_all.append(labels.numpy())
   if len(probs_all) == 0:
       return np.zeros((0, len(ALL_LABELS))), np.zeros((0, len(ALL_LABELS)))
   return np.vstack(probs_all), np.vstack(labels_all).astype(np.float32)
 
def tune_thresholds(probs_val, labels_val, steps=99):
   num_labels = labels_val.shape[1]
   thresholds = np.full(num_labels, 0.5, dtype=np.float32)
   per_class_f1 = np.zeros(num_labels, dtype=np.float32)
   for i in range(num_labels):
       unique_vals = np.unique(labels_val[:, i])
       if len(unique_vals) < 2:
           thresholds[i] = 0.5
           per_class_f1[i] = 0.0
           continue
       best_f1 = 0.0
       best_t = 0.5
       for t in np.linspace(0.01, 0.99, steps):
           f1 = f1_score(labels_val[:, i], (probs_val[:, i] > t).astype(int), zero_division=0)
           if f1 > best_f1:
               best_f1 = f1
               best_t = t
       thresholds[i] = best_t
       per_class_f1[i] = best_f1
   return thresholds, per_class_f1
 
def evaluate_with_thresholds(probs, labels, thresholds):
   preds = (probs > thresholds).astype(int)
   macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
   per_class_f1 = []
   for i in range(labels.shape[1]):
       unique_vals = np.unique(labels[:, i])
       if len(unique_vals) < 2:
           per_class_f1.append(0.0)
       else:
           per_class_f1.append(f1_score(labels[:, i], preds[:, i], zero_division=0))
   aucs = []
   for i in range(labels.shape[1]):
       if len(np.unique(labels[:, i])) < 2:
           continue
       try:
           aucs.append(roc_auc_score(labels[:, i], probs[:, i]))
       except:
           pass
   macro_auc = float(np.mean(aucs)) if len(aucs) > 0 else np.nan
   return float(macro_f1), float(macro_auc), np.array(per_class_f1)
 
 
def active_learning_pipeline(pool_df, val_df, test_df,
                            rounds=3, query_size=1000, init_size=5000,
                            batch_size=64, image_size=224, epochs=4,
                            max_unlabeled_to_score=5000,
                            use_focal=True, focal_gamma=2.0,
                            use_mixup=False, mixup_alpha=0.4,
                            device=None):
   """
   pool_df: initial training pool (will be split into labeled/unlabeled)
   val_df: held-out validation (for threshold tuning & model selection)
   test_df: final test set
   """
   if device is None:
       device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print("Using device:", device)
 
   
   train_transform = transforms.Compose([
       transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
       transforms.RandomHorizontalFlip(),
       transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
       transforms.ToTensor(),
       transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225]),
   ])
   
   eval_transform = transforms.Compose([
       transforms.Resize((image_size, image_size)),
       transforms.ToTensor(),
       transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225]),
   ])
 
   pool_dataset = ChestXrayDataset(pool_df, transform=train_transform)
   val_dataset = ChestXrayDataset(val_df, transform=eval_transform)
   test_dataset = ChestXrayDataset(test_df, transform=eval_transform)
 
   val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=DEFAULT_NUM_WORKERS)
   test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=DEFAULT_NUM_WORKERS)
 
   all_idx = np.arange(len(pool_dataset))
   np.random.shuffle(all_idx)
   labeled_idx = list(all_idx[:init_size])
   unlabeled_idx = list(all_idx[init_size:])
 
   model = MultiLabelResNet()
   model.to(device)
 
   
   def compute_pos_weight(indices):
       labeled_labels = np.vstack([pool_df.iloc[i][ALL_LABELS].values for i in indices]).astype(np.float32)
       N = labeled_labels.shape[0]
       class_counts = labeled_labels.sum(axis=0)
       neg = (N - class_counts)
       pos = class_counts
       pos_weight = (neg + 1.0) / (pos + 1.0)  
       pos_weight = np.clip(pos_weight, 1e-6, 1e6).astype(np.float32)
       return pos_weight, labeled_labels
 
   
   pos_weight_np, _ = compute_pos_weight(labeled_idx)
   pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32).to(device)
 
   
   if use_focal:
       criterion = FocalBCEWithLogits(alpha=1.0, gamma=focal_gamma, pos_weight=pos_weight)
   else:
       criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
 
   optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, verbose=True)
 
   best_val_f1 = -1.0
   best_state = None
 
   loss_history = []
   f1_history = []
   auc_history = []
 
   for r in range(rounds):
       print(f"\n=== ROUND {r+1}/{rounds} | labeled={len(labeled_idx)} ===")
 
       
       pos_weight_np, labeled_labels_matrix = compute_pos_weight(labeled_idx)
       pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32).to(device)
       if use_focal:
           criterion = FocalBCEWithLogits(alpha=1.0, gamma=focal_gamma, pos_weight=pos_weight)
       else:
           criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
 
       
       class_freq = (labeled_labels_matrix.sum(axis=0) / (labeled_labels_matrix.shape[0] + 1e-12)).astype(np.float32)
       inv_class_freq = (1.0 / (class_freq + 1e-6))
       inv_class_freq = inv_class_freq / (np.mean(inv_class_freq) + 1e-12)
       sample_weights_np = (labeled_labels_matrix * inv_class_freq).sum(axis=1)
       sample_weights_np = sample_weights_np + 0.1  
       sample_weights = torch.tensor(sample_weights_np, dtype=torch.float32)
       sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
 
       train_loader = DataLoader(Subset(pool_dataset, labeled_idx),
                                 batch_size=batch_size, sampler=sampler, num_workers=DEFAULT_NUM_WORKERS)
 
       
       for epoch in range(epochs):
           avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, use_mixup=use_mixup, mixup_alpha=mixup_alpha)
           print(f"Epoch {epoch+1} done, Loss={avg_loss:.4f}")
 
       
       probs_val, labels_val = predict_probs(model, val_loader, device)
       if probs_val.shape[0] == 0 or labels_val.shape[0] == 0:
           print("Validation set empty -- skipping threshold tuning.")
           thresholds = np.full(len(ALL_LABELS), 0.5, dtype=np.float32)
           val_macro_f1 = 0.0
           val_macro_auc = np.nan
           per_class_f1_val = np.zeros(len(ALL_LABELS), dtype=np.float32)
       else:
           thresholds, per_class_f1_val = tune_thresholds(probs_val, labels_val, steps=99)
           val_macro_f1, val_macro_auc, per_class_f1_metrics = evaluate_with_thresholds(probs_val, labels_val, thresholds)
           val_macro_f1 = float(val_macro_f1)
           val_macro_auc = float(val_macro_auc)
           print(f"Validation: f1={val_macro_f1:.4f} auc={val_macro_auc:.4f}")
 
       
       if val_macro_f1 > best_val_f1:
           best_val_f1 = val_macro_f1
           best_state = {
               'model': model.state_dict(),
               'thresholds': thresholds,
               'pos_weight': pos_weight_np.copy()
           }
           print("Saved new best model (by val F1).")
 
       scheduler.step(val_macro_f1)
 
       loss_history.append(avg_loss)
       f1_history.append(val_macro_f1)
       auc_history.append(val_macro_auc)
 
       
       if len(unlabeled_idx) == 0:
           print("No more unlabeled samples.")
           break
 
       if (max_unlabeled_to_score is not None) and (len(unlabeled_idx) > max_unlabeled_to_score):
           unlabeled_idx_sample = random.sample(unlabeled_idx, max_unlabeled_to_score)
           print(f"Unlabeled pool: {len(unlabeled_idx)}. Sampling {len(unlabeled_idx_sample)} for scoring to save time.")
       else:
           unlabeled_idx_sample = list(unlabeled_idx)
           print(f"Scoring entire unlabeled pool: {len(unlabeled_idx_sample)} samples.")
 
       unl_loader = DataLoader(Subset(pool_dataset, unlabeled_idx_sample),
                               batch_size=batch_size, shuffle=False, num_workers=DEFAULT_NUM_WORKERS)
 
       model.eval()
       scores = []
       with torch.no_grad():
           for imgs, _ in tqdm(unl_loader, desc="Scoring unlabeled", leave=False):
               imgs = imgs.to(device)
               probs = torch.sigmoid(model(imgs)).cpu().numpy()
               probs = np.clip(probs, 1e-8, 1-1e-8)
               entropy = - (probs * np.log(probs) + (1-probs) * np.log(1-probs))
               sample_entropy = np.sum(entropy, axis=1)
               scores.extend(sample_entropy)
 
       scores = np.array(scores)
       k = min(query_size, len(scores))
       if k == 0:
           print("No unlabeled samples available to query.")
           break
 
       selected_rel = np.argsort(scores)[-k:]
       selected_abs = [unlabeled_idx_sample[i] for i in selected_rel]
 
       labeled_idx.extend(selected_abs)
       unlabeled_idx = [u for u in unlabeled_idx if u not in selected_abs]
 
       print(f"Selected {len(selected_abs)} samples to add to labeled pool. New labeled size: {len(labeled_idx)}")
 
 
   if best_state is None:
       print("No best model saved, using current model and 0.5 thresholds.")
       final_thresholds = np.full(len(ALL_LABELS), 0.5, dtype=np.float32)
   else:
       model.load_state_dict(best_state['model'])
       final_thresholds = best_state['thresholds']
 
   probs_test, labels_test = predict_probs(model, test_loader, device)
   if probs_test.shape[0] == 0:
       test_macro_f1 = 0.0
       test_macro_auc = np.nan
       per_class_f1_test = np.zeros(len(ALL_LABELS), dtype=np.float32)
   else:
       test_macro_f1, test_macro_auc, per_class_f1_test = evaluate_with_thresholds(probs_test, labels_test, final_thresholds)
   print("\nFinal test metrics:")
   print(f"Test macro F1: {test_macro_f1:.4f}  AUC: {test_macro_auc:.4f}")
   print("Sample per-class F1 (first 10):", {ALL_LABELS[i]: float(per_class_f1_test[i]) for i in range(min(10, len(ALL_LABELS)))})
 
   # plots
   plt.figure(figsize=(10, 4))
   plt.plot(loss_history, marker='o')
   plt.title("Training Loss (avg per round)")
   plt.show()
 
   plt.figure(figsize=(10, 4))
   plt.plot(f1_history, marker='o')
   plt.title("Validation F1 per Round")
   plt.show()
 
   plt.figure(figsize=(10, 4))
   plt.plot(auc_history, marker='o')
   plt.title("Validation AUC per Round")
   plt.show()
 
   # Saving checkpoint
   save_dir = "saved_models"
   os.makedirs(save_dir, exist_ok=True)
 
   checkpoint = {
   "model_state_dict": best_state['model'] if best_state is not None else model.state_dict(),
   "thresholds": best_state['thresholds'] if best_state is not None else np.full(len(ALL_LABELS), 0.5, dtype=np.float32),
   "pos_weight": best_state['pos_weight'] if best_state is not None else pos_weight_np,
   "loss_history": loss_history,
   "val_f1_history": f1_history,
   "val_auc_history": auc_history,
   "ALL_LABELS": ALL_LABELS}
 
   save_path = os.path.join(save_dir, "final_model_checkpoint.pth")
   torch.save(checkpoint, save_path)
   print(f"\n✅ Training complete. Everything saved to {save_path}")
 
 
   return loss_history, f1_history, auc_history, test_macro_f1, test_macro_auc, per_class_f1_test
 
 
if __name__ == "__main__":
   csv_path = "/Users/devapriyansahayagoodwin/Documents/cost_final_code.py/Data_Entry_2017_v2020.csv"
   base_dir = "/Users/devapriyansahayagoodwin/Documents/cost_final_code.py/images/"
 
   print("Loading metadata...")
   df = load_metadata(csv_path)
   print("Encoding labels + verifying files...")
   df = encode_labels_and_paths(df, base_dir)
   print("Total valid images:", len(df))
 
   
   visualize_random_images(df)
   plot_label_distribution = lambda d: None
   
 
   
   train_pool_df, test_df = create_train_test(df, test_size=0.2, random_state=SEED)
 
   
   val_frac = 0.10
   if HAS_ML_STRAT:
       msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=SEED)
       X = train_pool_df.index.values.reshape(-1,1)
       Y = train_pool_df[ALL_LABELS].values
       tr_idx, val_idx = next(msss.split(X, Y))
       pool_df = train_pool_df.iloc[tr_idx].reset_index(drop=True)
       val_df = train_pool_df.iloc[val_idx].reset_index(drop=True)
   else:
       
       stratify_col = (train_pool_df[ALL_LABELS].values.sum(axis=1) > 0).astype(int)
       tr_idx, val_idx = train_test_split(np.arange(len(train_pool_df)), test_size=val_frac, random_state=SEED, stratify=stratify_col)
       pool_df = train_pool_df.iloc[tr_idx].reset_index(drop=True)
       val_df = train_pool_df.iloc[val_idx].reset_index(drop=True)
 
   print("Pool size (for active learning):", len(pool_df))
   print("Validation size:", len(val_df))
   print("Test size:", len(test_df))
 
   
   loss_hist, f1_hist, auc_hist, test_f1, test_auc, per_class_f1_test = active_learning_pipeline(
       pool_df, val_df, test_df,
       rounds=3,              
       query_size=1000,        
       init_size=5000,        
       batch_size=64,
       image_size=224,
       epochs=4,              
       max_unlabeled_to_score=5000,
       use_focal=True,        
       focal_gamma=2.0,
       use_mixup=False,        
       mixup_alpha=0.4,
       device=None
   )
 
   print("\nFinal Metrics:")
   print("Loss history:", loss_hist)
   print("Val F1 history:", f1_hist)
   print("Val AUC history:", auc_hist)
   print("Test macro F1:", test_f1)
   print("Test macro AUC:", test_auc)
