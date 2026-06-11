# Optimizare prin Gradient Descent pentru Regresia Liniară Multiplă

## Descriere
Acest proiect conține o implementare de la zero a algoritmului **Gradient Descent** (Coborârea pe gradient) în Python, utilizat pentru antrenarea unui model de Regresie Liniară Multiplă. Scriptul generează date sintetice, aplică algoritmul de optimizare pentru a găsi parametrii ideali (ponderile și bias-ul) și vizualizează procesul de învățare (scăderea funcției de cost).

## Funcționalități
* **Generare Date Sintetice**: Creează un set de date de test cu zgomot adăugat aleator, folosind un set predefinit de ponderi și un bias.
* **Calculul Funcției de Cost**: Implementează eroarea pătratică medie (Mean Squared Error - MSE) pentru a evalua performanța modelului la fiecare iterație.
* **Calculul Gradientului**: Determină derivatele parțiale ale funcției de cost în raport cu parametrii modelului.
* **Buclă de Optimizare**: Actualizează iterativ parametrii modelului pentru a minimiza costul.
* **Vizualizare**: Folosește `matplotlib` pentru a desena graficele costului în funcție de numărul de iterații, ilustrând convergența algoritmului.

---

## Concepte Matematice

Proiectul se bazează pe următoarele ecuații fundamentale:

**1. Modelul de predicție:**
Pentru o intrare cu trăsături multiple, predicția se calculează ca produsul scalar dintre vectorul de caracteristici și cel de ponderi, la care se adaugă bias-ul:
$$f_{\mathbf{w},b}(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b$$

**2. Funcția de Cost (J):**
Pentru a evalua acuratețea modelului pe un set de date cu $m$ exemple, folosim Eroarea Pătratică Medie:
$$J(\mathbf{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})^2$$

**3. Actualizarea Parametrilor (Gradient Descent):**
Parametrii sunt actualizați la fiecare pas folosind o rată de învățare $\alpha$:
$$w_j = w_j - \alpha \frac{\partial J}{\partial w_j}$$
$$b = b - \alpha \frac{\partial J}{\partial b}$$

---

## Cerințe
Pentru a rula acest proiect, ai nevoie de Python instalat (recomandat Python 3.7+) și de următoarele biblioteci externe:
* `numpy`
* `matplotlib`

Poți instala dependențele rulând:
> pip install numpy matplotlib

---

## Cum se rulează
1. Clonează acest repository sau descarcă fișierul `gradient_descent.py`.
2. Deschide un terminal în directorul în care se află fișierul.
3. Rulează scriptul:
> python gradient_descent.py

---

## Exemplu de Output

În consolă vei vedea progresul algoritmului la fiecare 10% din iterații:
> Iteration 0
> Cost    25.50
> dj_dW: [...]
> dJ_db: [...]
> w: [...], b:[...]

La final, scriptul va afișa parametrii găsiți de algoritm în comparație cu cei reali și va deschide o fereastră cu două grafice:
* **Cost vs. Iteration**: Evoluția funcției de cost pe toată durata antrenării.
* **Cost vs. Iteration (tail)**: O vedere detaliată (zoom) pe ultimele iterații pentru a observa convergența fină.