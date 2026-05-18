# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-05-18

### Added
- Initial release of TeleChurnIQ
- Machine Learning Pipeline
  - Random Forest model achieving 91.9% accuracy
  - XGBoost, Decision Tree, Logistic Regression for benchmarking
  - Comprehensive preprocessing and feature engineering
  - Model evaluation with multiple metrics (accuracy, precision, recall, F1-score, ROC-AUC)

- Explainability Layer
  - SHAP integration for feature contribution analysis
  - Custom narrative explanation engine
  - Global and local interpretability

- Frontend
  - React-based UI with Tailwind CSS
  - Landing page with project overview
  - Prediction interface with real-time scoring
  - Interactive KPI dashboard components
  - Responsive design with animations

- Backend
  - Express.js API server
  - Prediction endpoint `/api/predict`
  - CORS support for frontend communication
  - Error handling and validation

- Documentation
  - Comprehensive README.md with architecture diagrams
  - Model experiments documentation
  - Contributing guidelines
  - Setup and installation instructions

### Technical Stack
- **Backend:** Express.js, Node.js
- **Frontend:** React 19, Tailwind CSS 4, Vite
- **ML Service:** Python, scikit-learn, SHAP, XGBoost
- **Database:** CSV-based (IBM Telco dataset)

### Known Limitations
- Offline analysis only (real-time deployment ready but not implemented)
- Single dataset (IBM Telco - can be extended to other datasets)
- No automated model retraining (manual process)

## Planned for Future Releases

### [1.1.0] - Scheduled
- [ ] Batch prediction API for campaign targeting
- [ ] Automated model retraining pipeline
- [ ] Unit tests and integration tests
- [ ] Docker containerization
- [ ] Model performance monitoring and drift detection

### [2.0.0] - Planned
- [ ] Real-time streaming predictions
- [ ] What-if analysis for retention strategies
- [ ] Customer segmentation clustering
- [ ] Deep learning model integration (LSTM/GRU)
- [ ] Multi-tenancy support
- [ ] Cloud deployment (AWS/GCP/Azure)

### [3.0.0] - Long-term Vision
- [ ] Online learning with new predictions
- [ ] Causal inference analysis
- [ ] Cross-domain adaptation (other industries)
- [ ] Federated learning for collaborative training
- [ ] Advanced counterfactual explanations

---

## How to Report Changes

To propose new features or report changes:
1. Check the [Issues](https://github.com/yourusername/TeleChurnIQ/issues) page
2. Open a new issue or pull request
3. Reference this changelog if applicable
