from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def rocc(y_test, nb_score, dt_score, rf_score):

    fpr_nb, tpr_nb, _ = roc_curve(y_test, nb_score)
    fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_score)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_score)

    plt.plot(fpr_nb, tpr_nb, label=f'Naive Bayes (AUC={auc(fpr_nb,tpr_nb):.2f})')
    plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC={auc(fpr_dt,tpr_dt):.2f})')
    plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={auc(fpr_rf,tpr_rf):.2f})')

    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.show()
