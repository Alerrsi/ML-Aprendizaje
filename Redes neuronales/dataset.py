import matplotlib.pyplot as plt
import sklearn.datasets as ds

datos = ds.load_digits()
X = datos.data
Y = datos.target

print(f"X = {len(X)}")
print(f"Y = {len(Y)}")


for i in range(10):
    plt.subplot(2,5, i+1)
    plt.imshow(datos.images[i],cmap="gray")
    plt.title(datos.target[i])
    plt.axis("off")

plt.show()

