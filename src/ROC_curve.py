from naiveBayes import nb_model
from decTree import tree_model
from rForest import rfc_model
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def rocc(ch_data, lb):
    yt1,yp,ys1,x,le,md,acc= nb_model(ch_data, lb)
    yt2,yp,ys2,x,le,md,acc = tree_model(ch_data, lb)
    yt3,yp,ys3,x,le,md,acc= rfc_model(ch_data, lb)

    # ROC
    fpr_nb, tpr_nb, _ = roc_curve(yt1, ys1 )
    fpr_dt, tpr_dt, _ = roc_curve(yt2, ys2)
    fpr_rf, tpr_rf, _ = roc_curve(yt3, ys3)

    plt.plot(fpr_nb, tpr_nb, label=f'Naive Bayes (AUC={auc(fpr_nb,tpr_nb):.2f})')
    plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC={auc(fpr_dt,tpr_dt):.2f})')
    plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={auc(fpr_rf,tpr_rf):.2f})')

    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.show()
