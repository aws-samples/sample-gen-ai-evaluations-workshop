# Structured Data Evaluation Framework

This module is part of a larger Intelligent Document Processing (IDP) pipeline that handles document extraction, transformation, and evaluation. While the complete pipeline includes document preprocessing, OCR, field extraction, and post-processing, this framework specifically focuses on evaluating the accuracy of structured data extraction.

## Position in IDP Pipeline

```
Raw Document → (Optional) OCR → Field Extraction → Structured Evaluation → Results Analysis
                                                            ↑
                                                       This Framework
```

The evaluation framework serves as a critical quality control component by:
- Validating extraction accuracy against known ground truth
- Providing detailed metrics at both document and field levels
- Enabling systematic assessment of extraction quality
- Supporting continuous improvement of extraction models


## Next Steps

This workshop covered the basics of structured data evaluation. To explore the full capabilities of the IDP pipeline:

1. **Evaluation Framework**
   - For detailed documentation, visit the evaluation repository at [coming-soon].
   - Explore advanced comparison strategies
   - Review more complex examples in the examples directory

3. **Integration with IDP Pipeline**
   - For detailed documentation, visit the IDP accelerator repository at [repository-link](https://github.com/aws-solutions-library-samples/accelerated-intelligent-document-processing-on-aws).
   - Connect with upstream document processing modules
   - Choose from various patterns


