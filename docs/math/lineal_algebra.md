# Algebra Lineal

El álgebra lineal es una rama de las matemáticas que se utiliza ampliamente en la ciencia y la ingeniería. A diferencia de otras ramas de las matemáticas que se centran en números discretos, el álgebra lineal trabaja con conceptos continuos. Esto puede ser un desafío para muchos, ya que no están familiarizados con estos conceptos.

Sin embargo, una buena comprensión del álgebra lineal es crucial para entender y trabajar con muchos algoritmos de machine learning, especialmente en el campo del deep learning. Por lo tanto, es importante que nos familiaricemos con estas ideas para poder aplicar correctamente estas técnicas en futuros proyectos.

## Fundamentos
### Escalares, vectores, matrices y tensores 

- **Escalares:**  Es un solo número, a diferencia de la mayoría de los objetos estudiados en álgebra lineal, que suelen ser matrices de varios números. Estos se escriben en cursiva y en minúscula. Cuando los denotamos ademas se debe especificar qué tipo de número. Un ejemplo _$Sea\ s \in\ \mathbb{R}$, la pendiente de una recta_, es la definición de un escalar de valor real. 

```python
import numpy as np

# Escalar
s = np.float64(3.14)
print(f'Escalar: {s}') #3.14
print(s.shape) #()
```

- **Vectores:** En general, un vector es un objeto matemático especial que puede ser multiplicado por un escalar y/o multiplicado por otro vector, y el resultado siempre será un objeto del mismo tipo que el original. Una forma más concreta de ver esto es desde el enfoque de las ciencias de la computación, donde un vector se puede considerar como una matriz de números. Los números están dispuestos en un orden específico, y podemos identificar cada número individual por su índice en ese orden. Normalmente, los vectores se nombran con letras minúsculas y en negrita, como $\mathbf{x}$.
 
    Los elementos del vector se identifican escribiendo su nombre en cursiva, con un subíndice. El primer elemento de $x$ es $x_{1}$ , el segundo elemento es $x_{2}$ y así sucesivamente. 

    Si cada elemento está en $\mathbb{R}$, y el vector tiene n elementos, entonces el vector está en el conjunto formado por el producto cartesiano de $\mathbb{R}$, n veces, denominado $\mathbb{R^n}$.

$$\begin{align*}
    x &= \begin{bmatrix}
           x_{1} \\
           x_{2} \\
           \vdots \\
           x_{m}
         \end{bmatrix}
  \end{align*}$$

```python
import numpy as np

# Vector
vector = np.array([1, 2, 3, 4], dtype=np.float64)
print(f'Vector: {vector}') #[1. 2. 3. 4.]
print(vector.shape) #(4,)
```

- **Matrices:** Una matriz es una vector bidimensional de números, por lo que cada elemento se identifica por dos índices en lugar de sólo uno. Generalmente damos a las matrices nombres de variables en mayúsculas y en negrita, como $A$.

    Si una matriz de valores reales A tiene una altura de m y una anchura de n , decimos que $A\ \in\ \mathbb{R}_{m \times n}$ Normalmente identificamos los elementos de una matriz utilizando su nombre en cursiva, pero no en negrita, y los índices se enumeran con comas de separación. Por ejemplo, $a_{1,1}$, es la entrada superior izquierda de $A$ y $a_{m,n}$ es la entrada inferior derecha.

$$\begin{align*}
    A &= \begin{bmatrix}
      a_{1,1} & a_{1,2} & \cdots & a_{1,m} \\
      a_{2,1} & a_{2,2} & \cdots & a_{2,m} \\
      \vdots  & \vdots  & \ddots & \vdots  \\
      a_{m,1} & a_{m,2} & \cdots & a_{m,m}
    \end{bmatrix}
\end{align*}$$

```python
import numpy as np

# Matriz
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10,11,12]], dtype=np.float64)
print(f'Matriz:{matrix}')
'''[[ 1.  2.  3.]
   [ 4.  5.  6.]
   [ 7.  8.  9.]
   [10. 11. 12.]]'''
print(matrix.shape) #(4, 3)
```

- **Tensores:** Es el caso general, una matriz de números dispuestos en una cuadrícula regular con un número variable de ejes. Denotamos un tensor llamado $A$. Identificamos el elemento de $A$ en las coordenadas $( i, j, k )$ escribiendo $A_{i,j,k}$

```python
import numpy as np

# Tensor
tensor = np.array([[[1, 2, 3]], [[5, 6, 7]]], dtype=np.float64)
print(f'Tensor:{tensor}')
'''[
    [[1. 2. 3.]]
    [[5. 6. 7.]]
    ]
'''
print(tensor.shape) #(2,1,3)

```

## Multiplicación de matrices y vectores  

El producto de las matrices $A$ y $B$ es una tercera matriz $C$.Para que este producto esté definido, $A$ debe tener el mismo número de columnas que $B$ tiene en filas. Si $A$ tiene forma $m \times n$ y $B$ tiene forma $n \times p$ , entonces $C$ tiene forma $m \times p$. 

\begin{align*}
A = \begin{bmatrix}
a_{1,1} & a_{1,2} & \cdots & a_{1,n} \\
a_{2,1} & a_{2,2} & \cdots & a_{2,n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m,1} & a_{m,2} & \cdots & a_{m,n}
\end{bmatrix}_{m \times n}
\hspace{0.5cm}
B = \begin{bmatrix}
b_{1,1} & b_{1,2} & \cdots & b_{1,p} \\
b_{2,1} & b_{2,2} & \cdots & b_{2,p} \\
\vdots & \vdots & \ddots & \vdots \\
b_{n,1} & b_{n,2} & \cdots & b_{n,p}
\end{bmatrix}_{n \times p}
\end{align*}

El producto \( C = A \times B \) se calcula como:

\begin{align*}
C &= \begin{bmatrix}
c_{1,1} & c_{1,2} & \cdots & c_{1,p} \\
c_{2,1} & c_{2,2} & \cdots & c_{2,p} \\
\vdots & \vdots & \ddots & \vdots \\
c_{m,1} & c_{m,2} & \cdots & c_{m,p}
\end{bmatrix}_{m \times p}
\end{align*}

Se puede escribir el producto matricial simplemente colocando dos o más matrices juntas $C = AB$ donde la operación del producto esta definida como:

$$C_{i,j} = \sum_{k} A_{i,j}B_{i,j} $$

El producto punto entre dos vectores $x$ e $y$ de la misma dimensionalidad es el producto matricial $x^T y$. Se puede pensar en el producto matricial $C = AB$ como calculando $C_{i,j}$ como el producto punto entre la fila $i$ de $A$ y la columna $j$ de $B$.

**_Observación:_** El resultado del producto estándar de dos matrices no es una matriz que contiene el producto de los elementos individuales. Esta operación existe y se denomina producto Hadamard o element wise y es denotado por $A \odot B$

### Propiedades básicas

- La transposición de una matriz es la imagen espejada de la matriz a través de una línea diagonal, llamada diagonal principal, que discurre hacia abajo y a la derecha, a partir de su esquina superior izquierda. Se denota la transpuesta de la matrix $A$ como $A^T$

- Un escalar puede considerarse una matriz con un solo elemento. A partir de esto se puede ver que un escalar es su propia transposición, es decir, $a = a^T$.

- Se pueden sumar matrices entre sí, siempre que tengan la misma forma, simplemente sumando sus elementos correspondientes $C=A+B$ donde $C_{i,j} = A_{i,j} + B_{i,j}$.

- Se puede sumar un escalar a una matriz o multiplicar una matriz por un escalar, simplemente realizando esa operación en cada elemento de una matriz $D = a \cdot B + c$ donde
$D_{i,j} = a \cdot B_{i,j} + c$.

- El producto de matrices tiene diferentes propiedades. distributiva, asociativa, el transpuesto del producto y este no es conmutativa

$$Distributiva\ \ A(B+C) = AB + AC\\
Asociativa\ \ A(BC) = (AB)C\\
Transpuesto\ \ (AB)^T = B^T A^T$$

## Matriz identidad e inversas

- **Matriz identidad:** Es una matriz que no cambia ningún vector cuando se multiplica el vector por la matriz. Se denota la matriz identidad n dimensional como $I_{n}$. Formalmente $\forall x \in \mathbb{R}^n, I_nx = x$.

  La estructura de la matriz identidad es simple: todos los elementos de la diagonal principal son $1$ mientras que todos los otro elementos son $0$

$$\begin{align*}
 A &= \begin{bmatrix}
    1 & 0 & \cdots & 0 \\
    0 & 1 & \cdots & 0 \\
   \vdots  & \vdots  & \ddots & \vdots  \\
    0 & 0 & \cdots & 1
 \end{bmatrix}
\end{align*}$$

- **Matriz inversa:** La inversa de la matriz $A$ es denotada por $A^{-1}$ y esta definida por $$A^{-1} A = I_{n}$$
  
  Por supuesto, este proceso depende de que sea posible encontrar $A^{-1}$. Cuando $A^{-1}$ existe, existen varios algoritmos diferentes para encontrarla en forma cerrada.

  Utilizando la matriz inversa se puede resolver la ecuación lineal $Ax = b$ con el procedimiento

$$\begin{align*}
 Ax &= b\\
 A^{-1}Ax &= A^{-1}b\\
 I_{n}x &= A^{-1}b\\
 x &= A^{-1}b
\end{align*}$$

## Dependencia lineal y espacio vectorial (span)

- **Combinación lineal:** Es la suma de los productos de un escalar $c_i$ con un conjunto de vectores $\{v^{(1)}, v^{(2)}, \cdots ,v^{(i)}\}$, es decir, esta definido por la expresión 

$$\sum_i c_iv^{(i)} $$

$$
\begin{align*}
  \begin{pmatrix}
    20 \\
    12 \\
    37
  \end{pmatrix} = 
  
  2  \begin{pmatrix}
    1 \\
    3 \\
    5
  \end{pmatrix} +

  3  \begin{pmatrix}
    6 \\
    2 \\
    9
  \end{pmatrix}
\end{align*}$$

- **Dependencia lineal:** Es la redundancia de un vector entre dos vectores de una matriz. Formalmente se dice que un vector es linealmente dependiente si existe un vector que sea una combinación lineal de este. En caso de no existir se dice que el vector es linealmente independiente.

## Normas

La norma $L^p$ se define formalmente como:
$$ ||x||_p = \left( \sum_i |x_i|^p\right)^{\frac{1}{p}}\ para\ p \in \mathbb{R},p \geq 1$$

Las normas, incluida la norma L p, son funciones que asignan vectores a valores no negativos. A nivel intuitivo, la norma de un vector $x$ mide la distancia del origen al punto $x$. En términos más rigurosos, una norma es cualquier función $f$ que cumple las siguientes propiedades:

- $f(x) = 0 \implies x = 0$
- $f(x+y) \leq f(x) + f(y)$ (Desigualdad triangular)
- $\forall \alpha \in \mathbb{R}, f(\alpha x) = |\alpha|f(x)$

Un norma que surge comúnmente en machine learning es la norma $L^{\infty}$, también conocida como norma máxima. Esta norma se simplifica al valor absoluto del elemento con la mayor magnitud en el vector. $||x||_{\infty} = max_{i} |x_i|$

## Tipos especiales de matrices y vectores 

- **Matriz simétrica:** Es cualquier matriz que su transpuesta sea igual a la matriz $A = A^T$.
   
   Las matrices simétricas suelen surgir cuando los elementos se generan mediante alguna función de dos argumentos que no depende del orden de los argumentos. Por ejemplo si $A$ es una matriz de medidas de distancia, en la que $A_{i,j}$ da la distancia del punto $i$ al punto $j$, entonces $A_{i,j} = A_{j,i}$ porque las funciones de distancia son simétricas.

- **Vector unitario:**  Es cuando el vector tiene norma con valor igual a 1 $||x||_2 = 1$

- **Vectores ortogonales:** Dos vectores $x$, $y$ son ortogonales si $x^Ty = 0$. Si ambos vectores tienen norma distinta de cero, significa que forman un ángulo de $90$ grados entre sí. En $\mathbb{R}_n$, como máximo $n$ vectores pueden ser mutuamente ortogonales con norma distinta de cero. Si los vectores no sólo son ortogonales, sino que además tienen norma unitaria, se llaman ortonormales.

- **Matriz ortogonal:** Es una matriz cuadrada cuyas filas son y columnas son mutuamente ortonormales. $A^TA  = AA^T = I$. Esto implica que $A^{-1} = A^T$

## Descomposición en valores propios de una matriz

Eigendecomposition es una descomposición de una matriz en un conjunto de valor y vectores propios.

Sea $A$ una matriz cuadrada de orden $n \times n$ con $n$ vectores propios linealmente independientes $q_i$ donde $(i= 1, \cdots, n)$. Entonces $A$ puede ser factorizada como:

$$A = Q \Lambda Q^{-1} $$

donde $Q$ es la matriz cuadrada de orden $n \times n$ cuya columna i-ésima es el vector propio $q_i$ de $A$, y $\Lambda$ es la matriz diagonal cuyos elementos diagonales son los valores propios correspondientes, $\Lambda_{ii}= \lambda_i$.

$$\begin{align*}
Av &= \lambda v\\
AQ &= Q\Lambda \\
A &= Q\Lambda Q^{-1} 
\end{align*}$$

- Notar que las matrices diagonalizables se pueden factorizar de esta manera

## Descomposición en valores singulares (SVD)

Es una factorización de una matriz real o compleja. Generaliza la eigendecomposition de una matriz cuadrada normal con una base propia ortonormal a cualquier matriz $\ m\times n$.

Recordar que la Eigendecomposition consiste en analizar una matriz $A$ para descubrir una matriz $V$ de vectores propios y un vector de valores propios $\lambda$ tal que podamos reescribir $A$ como:

$$A = V  diag(\lambda) V^{-1}$$

La descomposición del valor singular es similar, excepto que esta vez se escribe $A$ como producto de tres matrices:

$$A = UDV^T $$

Suponiendo que $A$ es una matriz $m\times n$. Entonces $U$ se define como una matriz $m\times n$, D puede ser una matriz $m\times n$ y V puede ser una matriz $n\times n$. 

Se define que cada una de estas matrices tiene una estructura especial. Las matrices U y V se definen como matrices ortogonales. La matriz D se define como una matriz diagonal. Obsérvese que no es necesariamente que D sea cuadrada. Los elementos a lo largo de la diagonal de D se conocen como los valores singulares de la matriz $A$. Las columnas de $U$ se conocen como los vectores singulares izquierdos y las columnas de $V$ se conocen como vectores singulares derechos.

## Pseudoinverso de Moore-Penrose

La inversión de matrices no está definida para matrices que no son cuadradas. La pseudoinversa de se define como una matriz

$$A^+ = \lim_{\alpha \to 0}(A^TA+\alpha I)^{-1}A^T$$

Los algoritmos prácticos para calcular la pseudoinversa no se basan en esta definición, sino en la fórmula 

$$ A^+ = V D^+ U$$

Donde $U$,$D$ y $V$ son la descomposición en valores singulares de $A$, y el pseudoinverso $D^+$ de una matriz diagonal $D$ se obtiene tomando el recíproco de sus elementos no nulos
de sus elementos distintos de cero y luego tomando el transpuesto de la matriz resultante.

Cuando $A$ tiene más columnas que filas, la resolución de una ecuación lineal utilizando el pseudoinverso proporciona una de las muchas soluciones posibles. En concreto, se obtiene
la solución $x = A^+$ y con norma euclidiana mínima $||x||_2$ entre todas las soluciones posibles.

Cuando $A$ tiene más filas que columnas, es posible que no haya solución. En este caso, el uso de la pseudoinversa nos da la $x$ para la que $Ax$ es lo más cercano posible a y en términos de la norma euclidiana $||Ax - y||_2$

## Operador traza

El operador de traza da la suma de todas los elementos de las diagonales de una matriz.
$$ Tr(A) = \sum_i A_{i,j} $$

- Este operador es invariante a la transposición 

$$Tr(A) = Tr(A^T) $$

- La traza de una matriz cuadrada compuesta de muchos factores también es invariante a mover el último factor a la primera posición, si las formas de las correspondientes permiten definir el producto resultante 

$$Tr(ABC)=Tr(CAB)=Tr(BCA)$$

## Determinante

El determinante de una matriz cuadrada, denotado $det(A)$, es una función que asigna matrices a escalares reales. El determinante es igual al producto de todos los valores propios de la matriz. 

El valor absoluto del determinante puede considerarse como una medida de cuánto se expande o contrae el espacio. Si el determinante es 0, el espacio se contrae por completo en al menos una dimensión, perdiendo todo su volumen. Si el determinante es 1, entonces la transformación conserva el volumen.

## Referencias y recursos útiles
- [Deep Learning, Aaron Courville, Ian Goodfellow y Yoshua Bengio, Libro](https://www.deeplearningbook.org/)

- [Linear Algebra, Georgi E. Shilov, Libro](https://books.google.cl/books/about/Linear_Algebra.html?id=K-dQAAAAMAAJ&redir_esc=y)

- [Essence of linear algebra, 3Blue1Brown, Videos de YouTube](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)

- [Singular Value Decomposition (SVD), Videos de YouTube](https://www.youtube.com/watch?v=nbBvuuNVfco&ab_channel=SteveBrunton)