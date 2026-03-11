def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    def dif_quad(a, b, c, x):
        return 2*a*x + b
    x = float(x0)
    for i in range(steps):
        grad = dif_quad(a, b, c, x)
        x = x - lr*grad
    return float(x)
        
        
        