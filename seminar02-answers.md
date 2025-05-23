## Семинар 2

### Задача 1

$H_0: \sigma^2_1 = \sigma^2_2$ против $H_1: \sigma^2_1 > \sigma^2_2$

$F_{набл} = \frac{s^2_1}{s^2_2} = \frac{100}{25} = 4$

a) $F_{крит} = F(\text{df}_1 = 13, \text{df}_2 = 14, \alpha = 0.1) = 2.037$, в таблицах из задачника ближайшее $F(\text{df}_1 = 15, \text{df}_2 = 14, \alpha = 0.1) = 2.01$

критическая область $[2.037; +\infty)$, наблюдаемое значение лежит в ней, $H_0$ отвергается, первая дисперсия больше второй

b) $F_{крит} = F(\text{df}_1 = 13, \text{df}_2 = 14, \alpha = 0.05) = 2.507$, в таблицах из задачника ближайшее $F(\text{df}_1 = 15, \text{df}_2 = 14, \alpha = 0.05) = 2.463$

критическая область $[2.507; +\infty)$, наблюдаемое значение лежит в ней, $H_0$ отвергается, первая дисперсия больше второй

**NB.** Можно ли использовать здесь двустороннюю критическую область, если зафиксирована двусторонняя альтернатива? 
Да, можно, тогда критическая область будет иметь вид:

$$
[0; F(\text{df}_1, \text{df}_2, 1-\frac{\alpha}{2})] \cup [F(\text{df}_1, \text{df}_2, \frac{\alpha}{2}), +\infty)
$$

Однако если зафиксируем, что наблюдаемое значение F-статистики мы получаем делением *большей дисперсии на меньшую*, для вывода 
достаточно рассмотреть правую часть этой области $[F(\text{df}_1, \text{df}_2, \frac{\alpha}{2}), +\infty)$, что мы обычно и делаем, 
так как в таблицах представлены небольшие значения $\alpha$.

### Задача 2

a) $k = 5$

b) $n = 159$

c) $H_0: a_1  = \dots = a_5 = a$ против $H_1: \exists j \text{ } a_j \ne a$

d) $MSS = 2035.697$, $RSS = 2085.817$

$Mean\text{ } MSS = \frac{MSS}{\text{df}} = \frac{2035.697}{4} = 508.92$

$Mean\text{ } RSS = 13.54$

$F_{набл} = \frac{Mean\text{ } MSS}{Mean\text{ }RSS} = 37.57$

$F_{крит} = F(\text{df}_1 = 4, \text{df}_2 = 154, \alpha = 0.1) = 1.98$ (что точно, что по таблице при $\text{df}_2 = 160$)

критическая область $[1.98; +\infty)$, наблюдаемое значение лежит в ней, $H_0$ отвергается, не все средние равны

**NB.** Можем ли предположить двусторонюю критическую область и использовать для её построения $\frac{\alpha}{2}$, раз в альтернативной гипотезе всё равно стоит $\ne$? 
Нет, так как в отличие от случая сравнения дисперсий, здесь порядок деления при вычислении $F_{набл}$ изменить нельзя. Когда мы сравниваем две дисперсии с помощью F-критерия, 
мы можем большую дисперсию разделить на меньшую (получить значение выше 1, теоретически до $+\infty$), и меньшую на большую (получить значение ниже 1, 
теоретически от 0 до 1), отсюда возникает идея учёта двух хвостов распределения. А по смыслу и логике ANOVA необходимо делить среднюю 
сумму квадратов, отвечающую за межгрупповые различия, на среднюю сумму квадратов, отвечающую за внутригрупповые различия, никак не наоборот. 
Отсюда критической областью считается только диапазон значений выше $F_{крит} = F(\text{df}_1, \text{df}_2, \alpha)$, так как для отвержения 
нулевой гипотезы необходимо, чтобы значение F-статистики существенно превышало 1.

e) p-value ~ 0

f) $R^2 = \frac{MSS}{TSS} = \frac{2035.697}{2035.697 + 2085.817} \approx 0.49$, не очень высокое значение, 
вариацию баллов за КР на 49\% можно объяснить делением на группы.
