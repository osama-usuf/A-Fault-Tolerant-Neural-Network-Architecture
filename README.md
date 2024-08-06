# A Fault-Tolerant Neural Network Architecture

This repository contains an implementation of the fault-tolerant neural network architecture described in the paper "A Fault-Tolerant Neural Network Architecture" by Tao Liu et al. The implementation aims to mitigate weight disturbance problems in DNN accelerators without involving expensive retraining.

## Overview

The fault-tolerant neural network architecture proposed in the paper addresses performance issues in emerging DNN accelerators, particularly those based on resistive random access memory (ReRAM). Key components of this architecture include:

1. Collaborative logistic classifiers based on Error-Code Output Correction (ECOC)
2. Variable-length "decode-free" scheme
3. DNN-favorable searching code for codeword generation

This implementation focuses on enhancing the algorithmic error-resilience capability of DNN classifiers, making them more robust to potential hardware faults.

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- numpy
- scikit-learn

## Usage

1. Clone the repository
```git clone https://github.com/osama-usuf/A-Fault-Tolerant-Neural-Network-Architecture.git```

2. Install the required dependencies (ideally under a virtual environment)
```pip install -r requirements.txt```

3. Run the main script, see `argparse` definitions for argument descriptions
``` python3 main.py ```


## Implementation Details

The implementation follows these main steps:

1. Pre-train a simple network on MNIST
2. Extract the confusion matrix from the pre-trained model
3. Replace the softmax classifiers with collaborative logistic classifiers
4. Fine-tune the collaborative logistic classifiers using a custom loss function

The code includes implementations of:
- Collaborative Logistic Classifier
- Custom loss function for fine-tuning collaborative logistic classifiers
- Variable-length decode-free scheme
- Resistance variations (injected directly in the weight domain in the form of variations to network parameters)

The code does not implement following aspects:
- Pending zone logic (logistic classifiers always produce a single output `{0, 1}`)
- Stuck-at-fault (SAF) device defects (this will be included in a follow-up project with detailed device-level modeling)

## Contributing

Contributions to improve the implementation or extend it to other datasets and models are welcome. Please feel free to submit issues or pull requests.

## Citation

If you use this implementation in your research, please cite the original paper:

```
@inproceedings{liu2019fault,
  title={A fault-tolerant neural network architecture},
  author={Liu, Tao and Wen, Wujie and Jiang, Lei and Wang, Yanzhi and Yang, Chengmo and Quan, Gang},
  booktitle={Proceedings of the 56th Annual Design Automation Conference 2019},
  pages={1--6},
  year={2019}
}
```

## License

[MIT License](LICENSE)