import matplotlib.pyplot as plt

models = ['Random Forest', 'Naive Bayes', 'Combined Model']
accuracies = [69.20, 46.53, 69.20]

plt.figure(figsize=(8, 6))
plt.bar(models, accuracies, color=['green', 'blue', 'purple'])
plt.xlabel('Models')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy Comparison')
plt.ylim([0, 100])
plt.show()
