import numpy as np

# x1: 広さ, x2: 築年数, y: 価格
data = np.array([
    [51, 16, 3.0],
    [38,  4, 3.2],
    [57, 16, 3.3],
    [51, 11, 3.9],
    [53,  4, 4.4],
    [77, 22, 4.5],
    [63,  5, 4.5],
    [69,  5, 5.4],
    [72,  2, 5.4],
    [73,  1, 6.0]
])

x1 = data[:, 0]
x2 = data[:, 1]
y  = data[:, 2]

n = len(y)

print("データ概要")
print("データ数 n =", n)
print("説明変数: x1 = 広さ, x2 = 築年数")
print("目的変数: y = 価格")
print()

x1_bar = np.mean(x1)
x2_bar = np.mean(x2)
y_bar = np.mean(y)

dx1 = x1 - x1_bar
dx2 = x2 - x2_bar
dy = y - y_bar

Syy = np.sum(dy ** 2)
S11 = np.sum(dx1 ** 2)
S22 = np.sum(dx2 ** 2)
S12 = np.sum(dx1 * dx2)
S1y = np.sum(dx1 * dy)
S2y = np.sum(dx2 * dy)

Sxx = np.array([
    [S11, S12],
    [S12, S22]
])

Sxy = np.array([
    [S1y],
    [S2y]
])

beta = np.linalg.inv(Sxx) @ Sxy

B1 = beta[0, 0]
B2 = beta[1, 0]
B0 = y_bar - B1 * x1_bar - B2 * x2_bar

r_x1_y = S1y / np.sqrt(S11 * Syy)
r_x2_y = S2y / np.sqrt(S22 * Syy)

print("相関係数")
print("x1とyの相関係数 = {:.3f}".format(r_x1_y))
print("x2とyの相関係数 = {:.3f}".format(r_x2_y))
print()

phi_R = 2
phi_e = n - phi_R - 1
phi_T = n - 1

'''
y_hat = B0 + B1 * x1 + B2 * x2

y_hat_bar = np.mean(y_hat)

dy_hat = y_hat - y_hat_bar

Syy = np.sum(dy ** 2)
Syhat_yhat = np.sum(dy_hat ** 2)
Sy_yhat = np.sum(dy * dy_hat)
'''

SR = B1 * S1y + B2 * S2y
Se = Syy - SR
Ve = Se / phi_e

#R = Sy_yhat / np.sqrt(Syy * Syhat_yhat)
R2 = SR / Syy
R2_star = 1 - (Se / phi_e) / (Syy / phi_T)

#print()
#print("予測値")
#for i in range(n):
#    print(i + 1, y_hat[i])



# 例題3
print("例題3")

# x1だけを取り込んだMODEL1
B1_M1 = S1y / S11
B0_M1 = y_bar - B1_M1 * x1_bar

Se_M1_x1 = Syy - B1_M1 * S1y

phi_e_M1 = n - 1 - 1
F0_x1 = ((Syy - Se_M1_x1) / (phi_T - phi_e_M1)) / (Se_M1_x1 / phi_e_M1)

print("x1をモデルに取り込んだ場合")
print("Se(M1) = {:.3f}".format(Se_M1_x1))
print("F0 = {:.1f}".format(F0_x1))

# x2だけを取り込んだMODEL1
B2_M1 = S2y / S22
B0_M1_x2 = y_bar - B2_M1 * x2_bar

Se_M1_x2 = Syy - B2_M1 * S2y

F0_x2 = ((Syy - Se_M1_x2) / (phi_T - phi_e_M1)) / (Se_M1_x2 / phi_e_M1)

print()
print("x2をモデルに取り込んだ場合")
print("Se(M1) = {:.3f}".format(Se_M1_x2))
print("F0 = {:.2f}".format(F0_x2))

print()
print("F0が大きいx1をモデルに取り込む")

SR_M1 = Syy - Se_M1_x1
R2_M1 = SR_M1 / Syy
R2_star_M1 = 1 - (Se_M1_x1 / phi_e_M1) / (Syy / phi_T)

print()
print("MODEL1の推定式")
print("y = {:.3f} + {:.4f}x1".format(B0_M1, B1_M1))
print("R^2(M1) = {:.3f}".format(R2_M1))
print("R*^2(M1) = {:.3f}".format(R2_star_M1))

# 次にx2を追加するか検討
Se_M2 = Se
phi_e_M2 = phi_e

F0_add_x2 = ((Se_M1_x1 - Se_M2) / (phi_e_M1 - phi_e_M2)) / (Se_M2 / phi_e_M2)

print()
print("x2を追加する場合")
print("F0 = {:.1f}".format(F0_add_x2))

print()
print("F0 >= 2 より、x2もモデルに取り込む")

print()
print("MODEL2の推定式")
print("y = {:.3f} + {:.4f}x1 {:+.4f}x2".format(B0, B1, B2))
print("R^2(M2) = {:.3f}".format(R2))
print("R*^2(M2) = {:.3f}".format(R2_star))


# 例題4
print()
print("例題4")

y_hat = B0 + B1 * x1 + B2 * x2
e = y - y_hat
e_prime = e / np.sqrt(Ve)

Sxx_inv = np.linalg.inv(Sxx)

print("No. x1  x2   y   y_hat    e'    h")

for i in range(n):
    x_xbar = np.array([dx1[i], dx2[i]])

    D2 = (n - 1) * (x_xbar @ Sxx_inv @ x_xbar)
    h = 1 / n + D2 / (n - 1)

    print("{:2d}  {:2.0f}  {:2.0f}  {:.1f}  {:.3f}  {:+.2f}  {:.2f}".format(
        i + 1, x1[i], x2[i], y[i], y_hat[i], e_prime[i], h
    ))


# 例題5
print()
print("例題5")

x01 = 70
x02 = 10

y0_hat = B0 + B1 * x01 + B2 * x02

print("予測値")
print("y0_hat = {:.2f}".format(y0_hat))

x0_xbar = np.array([x01 - x1_bar, x02 - x2_bar])

D0_2 = (n - 1) * (x0_xbar @ Sxx_inv @ x0_xbar)

#print("D0^2 = {:.3f}".format(D0_2))

# phi_e = n - phi_R - 1 = 7
# t(7, 0.05) = 2.365
t_phie_005 = 2.365

conf_low = y0_hat - t_phie_005 * np.sqrt((1 / n + D0_2 / (n - 1)) * Ve)
conf_high = y0_hat + t_phie_005 * np.sqrt((1 / n + D0_2 / (n - 1)) * Ve)

pre_low = y0_hat - t_phie_005 * np.sqrt((1 + 1 / n + D0_2 / (n - 1)) * Ve)
pre_high = y0_hat + t_phie_005 * np.sqrt((1 + 1 / n + D0_2 / (n - 1)) * Ve)

print()
print("母回帰の95%信頼区間")
print("{:.2f}, {:.2f}".format(conf_low, conf_high))

print()
print("95%予測区間")
print("{:.2f}, {:.2f}".format(pre_low, pre_high))