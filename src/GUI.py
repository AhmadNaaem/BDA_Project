import tkinter as tk
from tkinter import messagebox, simpledialog, scrolledtext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from EDA import graphs, heatMap
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def launch_gui(accuracy, y_test, y_pred, X_test, label_encoders, rf, ch_data, original_data):
    root = tk.Tk()
    root.title("Visa Eligibility Predictor")
    root.geometry("400x200")
    root.configure(bg="#e6ffe6")  # Light green background

    acc_label = tk.Label(root, text=f"Accuracy: {accuracy:.2f}%", anchor='e', bg="#e6ffe6", fg="#006600")
    acc_label.place(relx=1.0, rely=0.0, anchor='ne')

    # --- Show EDA Report (on GUI) ---
    def show_eda_report():
        report = ""
        report += "HEAD:\n" + str(ch_data.head()) + "\n\n"
        import io, sys
        buffer = io.StringIO()
        ch_data.info(buf=buffer)
        report += "INFO:\n" + buffer.getvalue() + "\n"
        report += "UNIQUE VALUES:\n" + str(ch_data.nunique()) + "\n\n"
        report += "NULL VALUES:\n" + str(ch_data.isnull().sum()) + "\n\n"
        report += "DESCRIBE:\n" + str(ch_data.describe()) + "\n"
        eda_win = tk.Toplevel(root)
        eda_win.title("EDA Report")
        eda_win.configure(bg="#e6ffe6")
        txt = scrolledtext.ScrolledText(eda_win, width=100, height=30, bg="#e6ffe6", fg="#006600")
        txt.pack(fill=tk.BOTH, expand=True)
        txt.insert(tk.END, report)
        txt.config(state=tk.DISABLED)

    # --- Show EDA Visualization (as in encode) ---
    def show_eda_visualization():
        graphs(original_data)
        heatMap(ch_data)

    # --- Show Model Evaluation Report ---
    def show_model_eval():
        acc = accuracy
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()
        msg = f"Accuracy: {acc:.2f}%\n\nClassification Report:\n{report}"
        eval_win = tk.Toplevel(root)
        eval_win.title("Model Evaluation Report")
        eval_win.configure(bg="#e6ffe6")
        txt = scrolledtext.ScrolledText(eval_win, width=80, height=20, bg="#e6ffe6", fg="#006600")
        txt.pack(fill=tk.BOTH, expand=True)
        txt.insert(tk.END, msg)
        txt.config(state=tk.DISABLED)

    # --- Make Prediction (as in evalP) ---
    def make_prediction():
        input_cols = [
            'gender',
            'age',  # ask for age, not age_group
            'parental level of education',
            'grade',
            'extracurricular activities',
            'ielts_group',
            'financial sponsorship'
        ]

        form = tk.Toplevel(root)
        form.title("Enter Features for Prediction")
        form.configure(bg="#e6ffe6")
        entries = {}

        for idx, col in enumerate(input_cols):
            tk.Label(form, text=col, bg="#e6ffe6", fg="#006600").grid(row=idx, column=0, sticky='w', padx=5, pady=5)
            if col in label_encoders:
                categories = label_encoders[col].classes_
                var = tk.StringVar(form)
                var.set(categories[0])
                entry = tk.OptionMenu(form, var, *categories)
                entry.config(bg="#e6ffe6", fg="#006600", highlightbackground="#e6ffe6")
                entry.grid(row=idx, column=1, padx=5, pady=5)
                entries[col] = var
            else:
                entry = tk.Entry(form, bg="#e6ffe6", fg="#006600")
                entry.grid(row=idx, column=1, padx=5, pady=5)
                entries[col] = entry

        def submit():
            sample = []
            for col in input_cols:
                val = entries[col].get()
                sample.append(val)
            form.destroy()

            sample_df = pd.DataFrame([sample], columns=input_cols)
            # Convert age to age_group
            if 'age' in sample_df.columns:
                sample_df['age'] = sample_df['age'].astype(float)
                sample_df['age_group'] = np.where(sample_df['age'] < 18, 'Below 18',
                                        np.where(sample_df['age'] < 20, '18-19',
                                        np.where(sample_df['age'] < 22, '20-21',
                                        np.where(sample_df['age'] < 24, '22-23', '24 and above'))))
            # Encode categorical columns
            for col in label_encoders:
                if col in sample_df.columns:
                    sample_df[col] = label_encoders[col].transform(sample_df[col])
            # Reorder columns to match training data
            sample_df = sample_df[X_test.columns]
            pred = rf.predict(sample_df)
            if 'visa eligible' in label_encoders:
                pred_label = label_encoders['visa eligible'].inverse_transform(pred)[0]
            else:
                pred_label = pred[0]
            messagebox.showinfo("Prediction", f"Predicted 'visa eligible': {pred_label}")

        submit_btn = tk.Button(form, text="Submit", command=submit, bg="#009933", fg="white")
        submit_btn.grid(row=len(input_cols), column=0, columnspan=2, pady=10)

    # --- Buttons ---
    eda_btn = tk.Button(root, text="Show EDA Report", command=show_eda_report, bg="#009933", fg="white")
    eda_btn.pack(pady=10)

    eda_vis_btn = tk.Button(root, text="Show EDA Visualization", command=show_eda_visualization, bg="#009933", fg="white")
    eda_vis_btn.pack(pady=10)

    eval_btn = tk.Button(root, text="Show Model Evaluation Report", command=show_model_eval, bg="#009933", fg="white")
    eval_btn.pack(pady=10)

    pred_btn = tk.Button(root, text="Make Prediction", command=make_prediction, bg="#009933", fg="white")
    pred_btn.pack(pady=10)

    root.mainloop()