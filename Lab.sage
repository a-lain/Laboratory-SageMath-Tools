#!/usr/bin/python
# -*- coding: utf-8 -*-

# We import 'FuncFormatter' from 'matplotlib' library to be able to modify how the
# graphics axes are ticked.
from matplotlib.ticker import FuncFormatter
import itertools
import numpy as np
import copy
import scipy.stats


def format_number(x, latex=False, pre=3, signed=False, limsup=3, liminf=-3):
    r"""
    Returns a string representation of a number.

    Function that returns a string to represent a number (either in 'latex' format or
    in 'string' format) with a given precision. It can also be specified whether the
    '+' should appear before the number if the number is positive (the sign is
    always shown if the number is negative). If the decimal logarithm of the number
    is bigger than 'limsup' or smaller than 'liminf' scientific notation will be 
    used.

    INPUT:

    - x -- a number
    - latex -- whether two return a latex represenation (True) or a string
               representation (False). (default: False)
    - pre -- the number of significant digits. (default: 3)
    - signed -- whether the sign of the number should always be displayed (True) or 
                just if the number is negative (False). (default: False)
    - limsup -- if log10(x) is greater than this parameter, scientific notation will
                be used. (default: 3)
    - liminf -- if log10(x) is smaller than this parameter, scientific notation will
                be used. (default: -3)

    OUTPUT:

    A string that represents the number. When used in latex mode, this function does
    not return the opening and closing dollars $.

    EXAMPLES:

        format_number(273.5) --> "274"

        format_number(1004.3, latex=True) --> "1.00\𝚌𝚍𝚘𝚝10^{3}"
        format_number(1004.3, latex=False) --> "1.00*𝟷𝟶^(𝟹)"
        format_number(1004.3, latex=True, signed = True) --> "+𝟷.𝟶𝟶\𝚌𝚍𝚘𝚝𝟷𝟶^{𝟹}"
        
        format_number(1004.3, limsup=4) --> "1000"
        format_number(1004.3, limsup=3) --> "𝟷.𝟶𝟶*𝟷𝟶^(𝟹)"
        
        format_number(1004.3, pre=1) --> "𝟷*𝟷𝟶^(𝟹)"
        format_number(1004.3, pre=2) --> "𝟷.𝟶*𝟷𝟶^(𝟹)"
        format_number(1004.3, pre=5) --> "𝟷.𝟶𝟶𝟺𝟹*𝟷𝟶^(𝟹)"

    """
    x = float(x)
    # We check the inputs have the correct type; an exception is raised otherwise.
    if not isinstance(latex, bool):
        raise Exception(repr(latex) + " is not a boolean: 'True' o 'False'.")
    elif not isinstance(signed, bool):
        raise Exception(repr(signed) + " is not a boolean: 'True' o 'False'.")
    elif not (int(pre) == pre and pre>=0):
        raise Exception(repr(pre)+" is not a natural number ($\ge 0$).")
    elif int(limsup) != limsup:
        raise Exception(repr(limsup) + " is not an integer.")
    elif int(liminf) != liminf:
        raise Exception(repr(liminf) + " is not an integer.")
    else:
        pre = int(round(pre))
        limsup = int(round(limsup))
        liminf = int(round(liminf))
        # We calculate the decimal logarithm of 'x', unless it's zero. In that case,
        # we impose log(0):=0.
        logA = log(abs(x)) / log(float(10)) if abs(x)!=0 else 0
        # If we apply the floor function to 'logA' we will obtain the exponent needed
        # for the number's scientific notation. Please realize that the exponent is
        # computed even if the conditions for scientific notation display are not met;
        # this calculation will be needed later. Explanation: let our number 'x' be
        # greater than 1, for example between 10^3 and 10^4. Then the required 
        # exponent for the number's scientific notation is 3. On the contrary, if our
        # number 'x' is less than 1, for instance between 10^(-3) and 10^(-4), the
        # searched exponent will be -4. Therefore, the exponent will always be the
        # floor function of the 'logA'.
        # In order to avoid some rounding errors, we introduce the following:
        ex = int(round(logA)) if abs(round(logA) - logA) < 1e-10 else floor(logA)
        # If the conditions for scientific notation display are satisfied, we redefine
        # 'x' as old 'x' divided by 10 to the 'ex'.
        if logA>=limsup or logA<=liminf:
            x = float(x/10 ** ex)
            # The string formatter must be told how many decimals digits are wished
            # for the number representation. Since in scientific notation there is
            # always one digit before the decimal separtor, we want 'pre - 1'
            # decimal places.
            if signed:
                bas = '{:+{}f}'.format(float(x), "." + str(pre - 1))    
            else:
                bas='{:{}f}'.format(float(x), "." + str(pre - 1))
            if latex:
                return bas + "\cdot 10^{" + str(ex) + "}"
            else:
                return bas + "*10^(" + str(ex) + ")"
        else:
            # 'nc' is the number of decimal places we want our number representation
            # to have. If the exponent is greater than zero, we need as many decimals
            # as precision digits are left over after writing all digits before the
            # decimal point. To left of the decimal point there are exactly 'ex + 1'
            # digits, because the exponent of the units is 0, the exponent of the
            # decens is 1 and so on. Nevertheless, we may not have enough precision
            # digits to reach the decimal separator; in that case our previous 
            # formula would give us a negative number of decimal places. In order to
            # avoid that behaviour, we use the 'max' function. If the exponent is
            # negative, we want as may decimal digits after the last "left zero" as
            # precision digits we have. The position of the first decimal that is
            # nonzero is precisely '-ex'. So, starting in that position we want to
            # display 'pre' decimal digits. However, we are couting the decimal in
            # the position '-ex' twice; therefore we want '-ex + pre -1' decimal 
            # digits in total.
            nc = int(max(pre - ex - 1, 0) if ex>=0 else -ex+pre-1)
            if pre-ex-1 < 0:
                x = round(x, pre - ex - 1)
            if signed:
                num = '{:+{}f}'.format(float(x), "." + str(nc))
            else:
                num='{:{}f}'.format(float(x), "." + str(nc))
            return num


def cf(pre=3, signed=False, limsup=3, liminf=-3):
    r"""
    FuncFormatter function that uses 'format_number' to obtain the latex 
    representation of the ticks.
    
    INPUT:

    - pre -- the number of significant digits. (default: 3)
    - signed -- whether the sign of the number should always be displayed (True) or 
                just if the number is negative (False). (default: False)
    - limsup -- if log10(x) is greater than this parameter, scientific notation will
                be used. (default: 3)
    - liminf -- if log10(x) is smaller than this parameter, scientific notation will
                be used. (default: -3)

    OUTPUT:

    A FuncFormatter object.

    EXAMPLES:

        plot(-e^x, (x, 1, 10), tick_formatter=[cf(), cf()])

    """
    def cust_for(x, pos=None):
        return "$" + format_number(float(x), latex=True, pre=pre, signed=signed, 
            limsup=limsup, liminf=liminf) + "$"
    return FuncFormatter(cust_for)


def ticksCf(ticks, pre=3, signed=False, limsup=3, liminf=-3):
    r"""
    Applies cf to the provided ticks.

    INPUT:

    - ticks -- a list which contains the points to show on the graph.
    - pre -- the number of significant digits. (default: 3)
    - signed -- whether the sign of the number should always be displayed (True) or 
                just if the number is negative (False). (default: False)
    - limsup -- if log10(x) is greater than this parameter, scientific notation will
                be used. (default: 3)
    - liminf -- if log10(x) is smaller than this parameter, scientific notation will
                be used. (default: -3)

    OUTPUT:

    A FuncFormatter object.

    EXAMPLES:

        plot(-e^x, (x, 1, 10), tick_formatter=[ticksCf([0,1,7]), ticksCf([0,0.3,0.5])])

    """
    def cust_for(x, pos=None):
        if x in ticks:
            return "$" + format_number(float(x), latex=True, pre=pre, signed=signed, 
                limsup=limsup, liminf=liminf) + "$"
        else:
            return ""
    return FuncFormatter(cust_for)


def sigmaConf(n):
    r"""
    Returns the probability associated to a confidence of n times σ. That is,
    this returns the probability of -nσ<X<nσ where X is a normalized normal
    random variable.

    INPUT:

    - n -- a real positive number

    OUTPUT:

    A number between zero and one.

    """
    return scipy.stats.norm.cdf(n) - scipy.stats.norm.cdf(-n)

twosigma = sigmaConf(2)


class Magnitude:
    r"""
    This class is used to represent the value and the standard deviation
    of a physical magnitude, for example, of a measurement.

    It allows to calculate confidence intervals and to compute error propagation.

    METHODS:
    - Operations: +, -, *, / with scalars and Magnitudes and ** just for scalars.
                It follows the error propagation method.
    - dy : Returns the error at a specified confidence.
    - format : Returns a representation for the Magnitude (unicode or latex string).
    - Support for pretty_print, print and latex functions.
    - show : Calls print or pretty_print on itself but it allows for custom confidence alpha.
    
    MEMBERS:

    - val -- scalar. Normally represents the expected value of the physical magnitude.
    - sd -- standard deviation of the value. (default: 0)
    - df -- degrees of freedom of the associated random variable. (default: 0, meaning infinite)

    INPUT:

    - value -- scalar. Normally represents the expected value of the physical magnitude.
    - sd -- standard deviation of the value. (default: 0)
    - df -- degrees of freedom of the associated random variable. (default: 0, meaning infinite)


    OUTPUT:

    Returns an object of type Magnitude.

    EXAMPLES:

        Magnitude(100, 1, 4) --> object of type magnitude with value 100, standard deviation 1 and 4 degrees of freedom.
        Magnitude(100) --> object of type magnitude with value 100, standard deviation 0 and infinite degrees of freedom.
        Magnitude(100, 4) --> object of type magnitude with value 100, standard deviation 4 and infinite degrees of freedom.
        Magnitude(100, df=4, sd=1) --> object of type magnitude with value 100, standard deviation 1 and 4 degrees of freedom.


    """
    def __init__(self, value, sd=0, df=0):
        self.val = value
        self.sd = sd
        self.df = df

    def dy(self, alpha=twosigma):
        r"""
        Gives the error of the Magnitude with the desired confidence (alpha).

        If the Magnitude comes from a gaussian distribution, the coverage factor is
        calculated with the quantile function of a normal distribution. Otherwise,
        it is determined with the t-Student distribution instead.
        """
        if self.df == 0:
            return scipy.stats.norm.interval(alpha)[1] * self.sd
        else:
            return scipy.stats.t.interval(alpha, self.df)[1] * self.sd

    def __add__(self, other):
        try:
            ndf = min(filter(lambda x: x!=0, [self.df, other.df]))
        except:
            ndf = 0
        return Magnitude(self.val + other.val, sqrt(self.sd**2 + other.sd**2), ndf)

    def __radd__(self, other):
        return Magnitude(other + self.val, self.sd, self.df)

    def __sub__(self, other):
        try:
            ndf = min(filter(lambda x: x!=0, [self.df, other.df]))
        except:
            ndf = 0
        return Magnitude(self.val - other.val, sqrt(self.sd**2 + other.sd**2), ndf)

    def __mul__(self, other):
        try:
            float(other)
        except:
            try:
                ndf = min(filter(lambda x: x!=0, [self.df, other.df]))
            except:
                ndf = 0
            return Magnitude(self.val * other.val, sqrt(other.val**2*self.sd**2 + self.val**2*other.sd**2), ndf)
        else:
            return Magnitude(other*self.val, abs(other)*self.sd, self.df)

    def __rmul__(self, other):
        return Magnitude(other*self.val, abs(other)*self.sd, self.df)

    def __neg__(self):
        return (-1)*self

    def __truediv__(self, other):
        try:
            float(other)
        except:
            try:
                ndf = min(filter(lambda x: x!=0, [self.df, other.df]))
            except:
                ndf = 0
            return Magnitude(self.val / other.val, sqrt(self.sd**2 / other.val**2 + self.val**2/other.val**4*other.sd**2), ndf)
        else:
            return Magnitude(self.val / other, self.sd / abs(other), self.df)

    def __rtruediv__(self, other):
        return Magnitude(other / self.val, sqrt(other**2 / self.val**4 * self.sd**2), self.df)

    def __pow__(self, exponent):
        try:
            float(exponent)
        except:
            raise Exception("You can not raise a Magnitude to something that is not a number.")
        else:
            return Magnitude(self.val**exponent, exponent**2*self.val**(2*(exponent-1))*self.sd**2, self.df)

    def __eq__(self, other):
        return self.val == other.val

    def __lt__(self, other):
        return self.val < other.val

    def __le__(self, other):
        return self.val <= other.val

    def __gt__(self, other):
        return self.val > other.val

    def __ge__(self, other):
        return self.val >= other.val

    def format(self, alpha=twosigma, showError=True, useLatex=False):
        r"""
        Returns a representation of the Magnitude. 
        

        Finds the best representation of the value, according to its error. With the following convention: 
        The value must have its least significative digit at the same position as the first non-zero digit of 
        the associated error. If that first non-zero digit is 1 or 2, we also show the next digit; 
        also the corresponding one in the value.

        INPUT:

        - showError -- a boolean. Determines whether to show the error associated to
                        an confidence alpha.
        - alpha -- a value such that 0 <= x <= 1. Represents the confidence with
                   which you want to show the Magnitude.
        - latex -- a boolean. Determines whether to return a string representation
                      or a latex one.

        Returns an object of type Magnitude.

        EXAMPLES:

        We start with an object of type Magnitude like a = Magnitude(100, 1)

            a.format(alpha=0.683, showError=False, useLatex=False) --> 100.0
            a.format(alpha=0.683, showError=True, useLatex=True) --> 𝟷𝟶𝟶.𝟶\𝚙𝚖𝟷.𝟶
            a.format(0.683) --> 𝟷𝟶𝟶.𝟶±𝟷.𝟶
            a.format(alpha=0.683, showError=True, useLatex=False) --> 𝟷𝟶𝟶.𝟶±𝟷.𝟶
        
        """
        V = float(self.val)
        E = float(self.dy(alpha))
        if E != 0:
            # We calculate the difference in digits between the value and the error
            preV = floor(log(abs(V), 10)) - floor(log(abs(E), 10)) + 1 if V!=0 else 1
            # We get rif of the initial zeros of the numerical value of the error.
            w = repr(float(E)).replace(".", "")
            while w[0] == "0":
                w = w[1:]
            # The defualt precision is one significant digit.
            preE = 1
            # If the first digit of the error is a one or a two, we need one more
            # precision digit for both the error and the value.
            if (w[0]=="1" or w[0]=="2"):
                preE += 1
                preV += 1
            # If the current found precision for the value is negative, we force it to
            # be one.
            if preV <= 0:
                preV = 1
            if useLatex:
                if showError:
                    return format_number(V, pre=preV, latex=True) + r"\pm" +\
                        format_number(E, pre=preE, latex=True)
                else:
                    return format_number(V, pre=preV, latex=True)
            else:
                if showError:
                    return format_number(V, pre=preV) + u"\u00b1" +\
                        format_number(E, pre=preE)
                else:
                    return format_number(V, pre=preV)
        else:
            if useLatex:
                return format_number(V, pre=3, latex=True)
            else:
                return format_number(V, pre=3)

    def __repr__(self):
        r"""
        Returns a string representation.
        """
        return self.format()
    
    def _latex_(self):
        r"""
        Returns a latex representation.
        """
        return self.format(useLatex=True)

    def show(self, alpha=sigmaConf(twosigma), showError=True, useLatex=True):
        r"""
        Shows the Magnitude with its error (or not) either in latex representation or in unicode.

        Just complements the above function "format" with the pretty_print output for latex parameter to work.
        For more information read that function. The parameter showError allows the user to control
        whether the error is displayed (or not). A true value of the parameter useLatex makes the function
        return a latex representation. Otherwise, it returns a unicode string.  
        """
        if useLatex:
            pretty_print(LatexExpr(self.format(alpha, showError, useLatex)))
        else:
            print(self.format(alpha, showError, useLatex))


def plotErrorBars(Y, X, alpha=twosigma, errBarFactor=45, plotColor="blue"):
    r"""
    Returns a plot with the associated error bars for a pair of arrays of Magnitudes. 

    For a pair of arrays of Magnitudes; it draws the corresponding error bars for each
    point with the given confidence (alpha) and color (plotColor). The size of 
    these error bars automatically scales to the axes sizes.

    INPUT:

    -Y, X -- arrays of Magnitudes to represent and draw error bars.
    - alpha -- a value such that $0\le x \le 1$. Represents the uncertainty with
               which you want to show the Magnitude.
    - errBarFactor -- just a number that regulates the size of the edges of the 
                      error bars. The default value is set by trial and error.
    - plotColor -- string identifying a specific color for error bars.

    OUTPUT:

    Returns a graph with the error bars for each data in entry arrays.

    EXAMPLES:
    
        graph += plotErrorBars(Y, X, 0.683, plotColor = "orange") --> Adds orange 
                                    error bars to graph with one sigma confidence.
    """

    errLengthX = (max(X) - min(X)).val / errBarFactor
    errLengthY = (max(Y) - min(Y)).val / errBarFactor
    G = plot([])
    for x,y in list(zip(X, Y)):
        if x.sd>0:
            tx = scipy.stats.t.interval(alpha, x.df)[1] if x.df>=1 else scipy.stats.norm.interval(alpha)[1]
            Dx = tx*x.sd
            G += line2d([(x.val - Dx, y.val), (x.val + Dx, y.val)], color=plotColor)
            G += line2d([(x.val - Dx, y.val - errLengthY), (x.val - Dx, y.val + errLengthY)], color=plotColor)
            G += line2d([(x.val + Dx, y.val - errLengthY), (x.val + Dx, y.val + errLengthY)], color=plotColor)
        if y.sd>0:
            ty = scipy.stats.t.interval(alpha, y.df)[1] if y.df>=1 else scipy.stats.norm.interval(alpha)[1]
            Dy = ty*y.sd
            G += line2d([(x.val, y.val - Dy), (x.val, y.val + Dy)], color=plotColor)               
            G += line2d([(x.val - errLengthX, y.val - Dy), (x.val + errLengthX, y.val - Dy)], color=plotColor)
            G += line2d([(x.val - errLengthX, y.val + Dy), (x.val + errLengthX, y.val + Dy)], color=plotColor)
    return G



def plotMagnitudes(Y, X, pointsSize=40, alpha=twosigma, plotColor="blue", errBarFactor=45, 
            showErrorBars=True, tick_formatter=[cf(),cf()], *args, **kwargs):
    r"""
    Returns a plot for the Magnitude type given data. 

    This function is intended for adapting SAGE's list_plot() function to work directly 
    with Magnitudes.

    For two arrays of Magnitudes, plots the pairs of points corresponding to the values
    of the magnitudes and, if we choose to, plots the corresponding error bars. In addition,
    it sets axes to latex format. You can specify the plot color, the error bar scaling factor 
    and other parameters of the underlying functions (plotErrorBars).

    INPUT:

    - Y, X -- arrays of Magnitudes.
    - pointsSize -- point data size. Controls how big the data points are (default: 40).
    - alpha -- a value such that 0 <= x <= 1. Represents the confidence with
               which you want to show the Magnitude.
    - errBarFactor -- just a number that regulates the size of the edges of the 
                      error bars. The default value has been set by trial and error
                      and it is 45.
    - plotColor -- string identifying a specific color for the whole plot.
    - showErrorBars -- a boolean. Controls if you want error bars to show.
    - tick_formatter -- a matplotlib FuncFormatter. (default value makes axes ticks show in latex) 
    - *args
    - **kwargs

    OUTPUT:

    Returns a graph with the plot and error bars for the data in entry arrays.

    EXAMPLES:

        graph = plotMagnitudes(Y, X, pointsSize=10, plotColor="red", showErrorBars=False) --> 
                --> Now graph contains the plot of the entry data with a small point size, whitout error bars.
                    All in red.
    """
    if alpha >= 1 or alpha < 0:
            raise Exception("The value given for alpha is not a valid confidence.")
    G = plot([])
    G += list_plot(list(zip([x.val for x in X], [y.val for y in Y])), size=pointsSize, color=plotColor,
        tick_formatter=tick_formatter, *args, **kwargs)
    if showErrorBars:
        G += plotErrorBars(Y, X, alpha=alpha, errBarFactor=errBarFactor,
            plotColor=plotColor)
    return G


def tableMagnitudes(rows=None, columns=None, alpha=twosigma, showErrors=False, header_row=False, header_column=False, frame=True, align="center"):
    r"""

    This function allows to create tables where all the Magnitudes are shown with the same confidence, which can be selected
    by the user. If one were to use the normal table function, all Magnitudes will be expressed with a two sigma confidence.
    The arguments accepted by this function are the same ones as those accepted by original table function. For further information
    use help(table).

    """
    T = copy.deepcopy(rows) if rows!=None else copy.deepcopy(columns)
    for i in range(len(T)):
        for j in range(len(T[i])):
            T[i][j] = "$" + T[i][j].format(alpha, useLatex=True, showError=showErrors) + "$"
    if rows == None:
        return table(rows=None, columns=T, header_row=header_row, header_column=header_column, frame=frame, align=align)
    else:
        return table(rows=T, columns=None, header_row=header_row, header_column=header_column, frame=frame, align=align)


class MeasureStudent(Magnitude):
    def __init__(self, *values):
        self.data = values
        n = len(values)
        self.val = 1/n * sum(values)
        self.sd = sqrt(1 / (n - 1) / n * sum((X - self.val)**2 for X in values))
        self.df = n - 1



class Regression(object):
    r"""
    Class used to hold everything related to a regression of generic type. 
    Linear, gaussian...

    Groups different propierties that are usefull in a regression, as well as some
    functions with the objective of making the resulting object easy to show in the
    desired form, together. Graphic, showing fit resulting parameters with their errors, etc.


    METHODS:

    - yval -- returns the value of the image of a real number x according 
            to the model.
    - sy -- returns the standard deviation of the image of a real number x according 
            to the model.
    - dy -- returns the error of the image of a real number x according 
            to the model with a certain confidence alpha.
    - y -- the output is of type Magnitude. Returns the image of a real number x according 
            to the model.   
    - ymin -- takes as input the confidence factor alpha. Returns a function that 
            takes some value x as input and its output is the corresponding value 
            of the LOWER edge of the model confidence region.
    - ymax -- takes as input the confidence factor alpha. Returns a function that 
            takes some value x as input and its output is the corresponding value 
            of the UPPER edge of the model confidence region.
    - graphic -- gives a plot of the data, the fitting function and the confidence region of 
                the model.

    MEMBERS:

    - Y, X -- the arrays implied in the regression (of type Magnitude).
    - data -- list of the pairs of Magnitudes (x,y).
    - n -- number of components of Y or X (it must be the same).
    - df --  degrees of freedom of the model.
    - f -- model function for the fitting. 
    - param -- variables on which f depends.
    - grad -- array that holds the gradient of f function.
    - hessian -- Hessian matrix of f.
    - value -- maximun likelihood estimators of the model parameters, according to the data.
    - cov -- covariance matrix of the parameters estimators.
    

    INPUT:

    - Y, X -- the arrays implied in the regression (of type Magnitude).
    - f -- symbolic function that data will fit to.
    - param -- array of variables that f depends on.
    - a0 -- array of initial guesses to take as a start point for the algorithms.

    OUTPUT:

    Returns a Regression type object.
     
    EXAMPLES:
    
    If f depends on (x, a, b, c), that is f(x,a,b,c) and we have two arrays of Magnitudes, A, B:

        a = Regression(A, B, f, [x,a,b,c], [x0,a0,b0,c0]) # Where the x0,a0,b0,c0 are real numbers.

    """
    def __init__(self, Y, X, f, param, a0):
        self.Y = Y
        self.X = X
        self.data = list(zip([x.val for x in X], [y.val for y in Y]))
        self.n = n = len(X)
        self.df = n - len(a0)
        self.f = f
        self.param = param
        self.grad = [diff(f,x) for x in avars]
        self.hessian = [[diff(f,x,y) for y in avars] for x in avars]
        self.value = a0 # To do 
        self.cov = a0 # To do
        self.yval = lambda x: f(x, *self.value)

    def sy(self, x):
        r"""

        Returns the standard deviation of the image of a real number x according 
        to the model.

        """
        gradf = np.array([ff(x, *[a.val for a  in self.value]) for ff in self.grad])
        return sqrt(sum(sum(self.cov[i,j]*gradf[i]*gradf[j] for i in range(len(gradf))) for j in range(len(gradf))))

    def dy(self, x, alpha=twosigma):
        r"""

        Returns the error of the image of a real number x according 
        to the model with a certain confidence alpha.
        
        """
        if alpha >= 1 or alpha < 0:
            raise Exception("The value given for alpha is not a valid confidence.")
        t = scipy.stats.t.interval(alpha, self.df)[1]
        return t*self.sy(x)

    def y(self, x):
        r"""

        The output is of type Magnitude. Returns the image of a real number x according 
        to the model.   
        
        """
        return Magnitude(self.yval(x), self.sy(x), self.df)

    def ymin(self, alpha=twosigma):
        r"""

        Takes as input the confidence factor alpha. Returns a function that 
        takes some value x as input and its output is the corresponding value 
        of the LOWER edge of the model confidence region.
        
        """
        t = scipy.stats.t.interval(alpha, self.df)[1]
        return lambda x: self.yval(x) - t*self.sy(x)

    def ymax(self, alpha=twosigma):
        r"""

        Takes as input the confidence factor alpha. Returns a function that 
        takes some value x as input and its output is the corresponding value 
        of the UPPER edge of the model confidence region.
        
        """
        t = scipy.stats.t.interval(alpha, self.df)[1]
        return lambda x: self.yval(x) + t*self.sy(x)

    def graphic(self, pointsSize=40, alpha=twosigma, plotColor="blue", fillAlpha=0.25, errBarFactor=45, 
            showDataPoints=True, showErrorBars=True, showConfidenceRegion=True,
            legendPlot=None, xmin=None, xmax=None, tick_formatter=[cf(),cf()], *args, **kwargs):
        r"""

        Gives a plot of the data, the fitting function and the confidence region of 
        the model.

        INPUTS

        - pointsSize -- point data size. Controls how big the data points are (default: 40).
        - alpha -- a value such that 0 <= x <= 1. Represents the confidence with
                   which you want to show the Magnitude.
        - plotColor -- string identifying a specific color for the whole plot.
        - fillAlpha -- a number between zero and one. It determines the transparency of the confidence region. (default: 0.25)
        - errBarFactor -- just a number that regulates the size of the edges of the 
                          error bars. The default value has been set by trial and error
                          and it is 45.
        - showDataPoints -- a boolean. Controls whether to show the data used for the model or not.
        - showErrorBars -- a boolean. Controls if you want error bars to show.
        - showConfidenceRegion -- a boolean. Controls whether to show the confidence region of the model or not.
        - legendPlot -- a string. Legend of the plot.
        - xmin -- the minimum value of x. By default it holds the minimum value of the x coordinate of the data.
        - xmax -- the maximum value of x. By default it holds the maximum value of the x coordinate of the data.
        - tick_formatter -- a matplotlib FuncFormatter. (default value makes axes ticks show in latex).
        - *args
        - **kwargs

        OUTPUT

        A plot.

        """
        if alpha >= 1 or alpha < 0:
            raise Exception("The value given for alpha is not a valid confidence.")
        G = plot([], tick_formatter=[cf(),cf()], *args, **kwargs)
        if showDataPoints:
            G += plotMagnitudes(self.Y, self.X, pointsSize=pointsSize, alpha=alpha, 
                plotColor=plotColor, errBarFactor=errBarFactor, showErrorBars=showErrorBars)

        # Plot of the function and its confidence region
        # Ιf the user does not provide concrete xmax and xmin, we calculate then based on the data.
        xmin = min(self.X).val if xmin==None else xmin
        xmax = max(self.X).val if xmax==None else xmax
        if legendPlot == None:
            legendPlot = r"$\chi^2_{" + str(self.df) + r"}=" + format_number(self.testchi2, latex=True) +\
                r"\;\;p=" + format_number(self.pvalue, latex=True) + "$"
        G += plot(self.yval, xmin, xmax, linestyle="-.", color=plotColor, legend_label=legendPlot)
        if showConfidenceRegion:
            G += plot(self.ymin(alpha), xmin, xmax, linestyle="--", color=plotColor)
            G += plot(self.ymax(alpha), xmin, xmax, linestyle="--", fill=self.ymin(alpha), color=plotColor, fillcolor=plotColor, fillalpha=fillAlpha)
        return G


class RegressionErrorsOnlyInY(Regression):
    r"""
    Class used to hold everything related to a regression; but your regression will only take 
    into account errors in Y axis. Inherits from Regression class.  

    For more information look up Regression class.


    METHODS:

    For more information look up Regression class.

    MEMBERS:

    - Y, X -- the arrays implied in the regression (of type Magnitude).
    - data -- list of the pairs of Magnitudes (x,y).
    - n -- number of components of Y or X (it must be the same).
    - df --  degrees of freedom of the model.
    - f -- model function for the fitting. 
    - param -- variables on which f depends.
    - grad -- array that holds the gradient of f function.
    - hessian -- Hessian matrix of f.
    - value -- maximun likelihood estimators of the model parameters, according to the data.
    - cov -- covariance matrix of the parameters estimators.
    - testchi2 -- contains the resulting value of the Chi-Square goodness of fit test.
    - pvalue -- contains the p-value associated with the Chi-Square test.

    INPUT:

    - Y, X -- the arrays implied in the regression (of type Magnitude).
    - f -- symbolic function that data will fit to.
    - param -- array of variables that f depends on.
    - a0 -- array of initial guesses to take as a start point for the algorithms.

    OUTPUT:

    Returns a RegressionErrorsOnlyInY type object.
     
    EXAMPLES:
    
    If f depends on (x, a, b, c), that is f(x,a,b,c) and we have two arrays of Magnitudes, A, B:

        a = RegressionErrorsOnlyInY(A, B, f, [x,a,b,c], [x0,a0,b0,c0]) #Where the x0,a0,b0,c0 are numbers

    """
    def __init__(self, Y, X, f, param, a0):
        self.Y = Y
        self.X = X
        self.data = list(zip([x.val for x in X], [y.val for y in Y]))
        self.n = n = len(X)
        self.df = n - len(a0)
        self.param = param
        xvar = list(filter(lambda x: x not in param, f.variables()))[0]
        self.f = fast_float(f, *([xvar] + param))
        self.grad = [fast_float(diff(f,x), *([xvar] + param)) for x in param]
        minusl = lambda par: sum((y.val - self.f(x.val, *par))**2 / (2*y.sd*y.sd) for x,y in zip(X,Y))
        self.hessian = [[fast_float(diff(f,x,y), *([xvar] + param)) for y in param] for x in param]
        a = minimize(minusl, a0, algorithm='simplex')
        hessianl = np.matrix([[float(sum(-1/y.sd/y.sd * self.grad[k](x.val, *a) * self.grad[j](x.val, *a) +
            (y.val - self.f(x.val, *a))/y.sd/y.sd * self.hessian[j][k](x.val, *a) for x,y in zip(X,Y)))
            for k in range(len(self.grad))] for j in range(len(self.grad))])
        self.cov = np.linalg.inv(-hessianl)
        for i in range(len(a0)):
            if self.cov[i,i]<0:
                raise Exception("Negative variances!")
        self.value = [Magnitude(a[j], sqrt(self.cov[j,j]), self.df) for j in range(len(a))]
        self.yval = lambda x: self.f(x, *[a.val for a in self.value])
        self.testchi2 = float(sum((y.val - self.yval(x.val))**2 / y.sd / y.sd for x,y in zip(X,Y)))
        self.pvalue = scipy.stats.chi2.sf(self.testchi2, self.df)


class RegressionNoErrors(Regression):
    r"""
    Class used to hold everything related to a regression; but your regression will not any errors into account.
    Inherits from Regression class.  


    In this case the error of the data is estimated to achieve the best fit, so it
    is used as another parameter of fitting instead of using the data errors.
    For more information look up Regression class.


    METHODS:

    For more information look up Regression class.


    MEMBERS:

    - Y, X -- the arrays implied in the regression (of type Magnitude).
    - data -- list of the pairs of Magnitudes (x,y).
    - n -- number of components of Y or X (it must be the same).
    - df --  degrees of freedom of the model.
    - f -- model function for the fitting. 
    - param -- variables on which f depends.
    - grad -- array that holds the gradient of f function.
    - hessian -- Hessian matrix of f.
    - value -- maximun likelihood estimators of the model parameters, according to the data.
    - cov -- covariance matrix of the parameters estimators.


    INPUT:

    - Y, X -- the arrays implied in the regression (of type Magnitude).
    - f -- symbolic function that data will fit to.
    - param -- array of variables that f depends on.
    - a0 -- array of initial guesses to take as a start point for the algorithms.

    OUTPUT:

    Returns a Regression type object.
     
    EXAMPLES:
    
    If f depends of (x, a, b, c), that is f(x,a,b,c) and we have two arrays of Magnitudes, A, B:

        a = RegressionNoErrors(A, B, f, [x,a,b,c], [x0,a0,b0,c0]) #Where the x0,a0,b0,c0 are numbers

    """
    def __init__(self, Y, X, f, param, a0):
        self.Y = Y
        self.X = X
        self.data = list(zip([x.val for x in X], [y.val for y in Y]))
        self.n = n = len(X)
        self.df = n - len(a0)
        self.param = param
        xvar = list(filter(lambda x: x not in param, f.variables()))[0]
        self.f = fast_float(f, *([xvar] + param))
        self.grad = [fast_float(diff(f,x), *([xvar] + param)) for x in param]
        minusl = lambda par: sum((y.val - self.f(x.val, *par))**2 for x,y in zip(X,Y))
        self.hessian = [[fast_float(diff(f,x,y), *([xvar] + param)) for y in param] for x in param]
        a = minimize(minusl, a0, algorithm='simplex')
        sigma2 = 1/n * sum((y.val - self.f(x.val, *a))**2 for x,y in zip(X,Y))
        hessianl = np.matrix([[float(sum(-1/sigma2 * self.grad[k](x.val, *a) * self.grad[j](x.val, *a) +
            (y.val - self.f(x.val, *a))/sigma2 * self.hessian[j][k](x.val, *a) for x,y in zip(X,Y)))
            for k in range(len(self.grad))] for j in range(len(self.grad))])
        self.cov = np.linalg.inv(-hessianl)
        for i in range(len(a0)):
            if self.cov[i,i]<0:
                raise Exception("Negative variances!")
        self.value = [Magnitude(a[j], sqrt(self.cov[j,j]), self.df) for j in range(len(a))]
        self.yval = lambda x: self.f(x, *[a.val for a in self.value])

    def graphic(self, pointsSize=40, alpha=twosigma, plotColor="blue", fillAlpha=0.25, errBarFactor = 45, 
        showErrorBars=True, showConfidenceRegion=True, tick_formatter=[cf(),cf()], *args, **kwargs):
        return super().graphic(pointsSize=pointsSize, alpha=alpha, plotColor=plotColor, 
            fillAlpha=fillAlpha, errBarFactor=errBarFactor, showErrorBars=showErrorBars, 
            showConfidenceRegion=showConfidenceRegion, tick_formatter=[cf(),cf()],
            legendPlot="", *args, **kwargs)


class LinearRegression(Regression):
    r"""
    Class used to hold everything related to a linear regression; taking into account Y 
    and X data errors.
    Inherits from Regression class.  


    In this particular case, the fitting is calculated with Newton-Raphson algorithm, taking as 
    starting variables the results of a linear regression with only errors in Y, that has a precise
    analytic solution.
    For more information look up Regression class.


    METHODS:

    For more information look up Regression class.


    MEMBERS:

    - Y, X -- the arrays implied in the regression (of type Magnitude).
    - data -- list of the pairs of Magnitudes (x,y).
    - n -- number of components of Y or X (it must be the same).
    - df --  degrees of freedom of the model.
    - a -- the slope.
    - b -- the y-intercept.
    - value -- maximun likelihood estimators of the model parameters, according to the data.
    - cov -- covariance matrix of the parameters estimators.
    - testchi2 -- contains the resulting value of the Chi-Square goodness of fit test.
    - pvalue -- contains the p-value associated with the Chi-Square test.

    INPUT:

    - Y, X -- the arrays implied in the regression (of type Magnitude).
    - tol -- controls the degree of precision for Newton-Raphson algorithm.
    - IterMax -- sets the maximum number of iterations for Newton-Raphson.

    OUTPUT:

    Returns a Regression type object.
     
    EXAMPLES:
    
        a = LinearRegression(A, B) 
        a = LinearRegression(A, B, 1e-7, 10000)

    """
    def __init__(self, Y, X, tol=1e-6, IterMax=1000):
        if len(X) != len(Y):
            raise Exception("Length of X is different from Length of Y")
        self.Y = Y
        self.X = X
        self.data = list(zip([x.val for x in X], [y.val for y in Y]))
        self.n = n = len(X)
        self.df = n - 2

        def g(a):
            Chi2s = [a*a*x.sd*x.sd + y.sd*y.sd for x,y in zip(X,Y)]
            b = sum((y.val - a*x.val) / chi2 for x,y,chi2 in zip(X,Y,Chi2s)) / sum(1/chi2 for chi2 in Chi2s)
            Epsilons = [y.val - a*x.val - b for x,y in zip(X,Y)]
            return -a*sum(x.sd*x.sd / chi2 * (1 - eps*eps / chi2) for x,eps,chi2 in zip(X, Epsilons, Chi2s)) +\
                sum(x.val*eps / chi2 for x,eps,chi2 in zip(X, Epsilons, Chi2s))
        
        def dgda(a):
            Chi2s = [a*a*x.sd*x.sd + y.sd*y.sd for x,y in zip(X,Y)]
            b = sum((y.val - a*x.val) / chi2 for x,y,chi2 in zip(X,Y,Chi2s)) / sum(1/chi2 for chi2 in Chi2s)
            Epsilons = [y.val - a*x.val - b for x,y in zip(X,Y)]
            return sum(1 / chi2 * (-x.sd*x.sd - x.val*x.val + (eps*eps + 2*a*a*x.sd*x.sd - 4*a*x.val*eps)*x.sd*x.sd/chi2 
                - 4*a*a*eps*eps*x.sd**4/chi2/chi2 ) for x,y,eps,chi2 in zip(X,Y,Epsilons,Chi2s) )

        # Newton Raphson
        a0 = LinearRegressionErrorsOnlyInY(Y, X).a.val
        a1 = a0 - g(a0) / dgda(a0)
        i = 0
        while (i < IterMax):
            a0 = a1
            a1 = a0 - g(a0) / dgda(a0)
            if (abs(g(a1)) < tol):
                break
        a = a1

        # New operations
        Chi2s = [a*a*x.sd*x.sd + y.sd*y.sd for x,y in zip(X,Y)]
        b = sum((y.val - a*x.val) / chi2 for x,y,chi2 in zip(X,Y,Chi2s)) / sum(1/chi2 for chi2 in Chi2s)
        Epsilons = [y.val - a*x.val - b for x,y in zip(X,Y)]
        cov = -sum(x.val / chi2 + 2*a*eps*x.sd*x.sd/chi2/chi2 for x,eps,chi2 in zip(X,Epsilons,Chi2s))
        H = np.matrix([[float(dgda(a)), float(cov)], [float(cov), float(-sum(1/chi2 for chi2 in Chi2s))]])
        self.cov = np.linalg.inv(-H)
        self.value = [Magnitude(a, sqrt(self.cov[0,0]), self.df), Magnitude(b, sqrt(self.cov[1,1]), self.df)]
        self.a = self.value[0]
        self.b = self.value[1]
        self.yval = lambda x: a*x + b
        self.sy = lambda x: sqrt(x*x*self.cov[0,0] + self.cov[1,1] + 2*x*self.cov[0,1])
        self.testchi2 = float(sum((y.val - self.yval(x.val))**2 / chi2 for x,y,chi2 in zip(X,Y,Chi2s)))
        self.pvalue = scipy.stats.chi2.sf(self.testchi2, self.df)


class LinearRegressionErrorsOnlyInY(Regression):
    r"""
    Class used to hold everything related to a linear regression; taking into account only Y data 
    errors.
    Inherits from Regression class.  

    In this particular case, the fitting is computed with the analityc solution of a linear 
    regression with the errors in Y following a gaussian distribution.
    For more information look up Regression class.


    METHODS:

    For more information look up Regression class.

    MEMBERS:

    - Y, X -- the arrays implied in the regression (of type Magnitude).
    - data -- list of the pairs of Magnitudes (x,y).
    - n -- number of components of Y or X (it must be the same).
    - df --  degrees of freedom of the model.
    - a -- the slope.
    - b -- the y-intercept.
    - value -- maximun likelihood estimators of the model parameters, according to the data.
    - cov -- covariance matrix of the parameters estimators.
    - testchi2 -- contains the resulting value of the Chi-Square goodness of fit test.
    - pvalue -- contains the p-value associated with the Chi-Square test.

    INPUT:

    - Y, X -- arrays of magnitudes to fit.

    OUTPUT:

    Returns a Regression type object.
     
    EXAMPLES:

        a = LinearRegressionErrorsOnlyInY(A, B) 

    """
    def __init__(self, Y, X):
        if len(X) != len(Y):
            raise Exception("Length of X is different from Length of Y")
        self.Y = Y
        self.X = X
        self.data = list(zip([x.val for x in X], [y.val for y in Y]))
        self.n = n = len(X)
        self.df = n - 2
        Sx = sum(x.val / y.sd / y.sd for x,y in zip(X,Y))
        Sy = sum(y.val / y.sd / y.sd  for y in Y)
        Sxx = sum(x.val**2 / y.sd / y.sd  for x,y in zip(X,Y))
        Sxy = sum(x.val*y.val / y.sd / y.sd  for x,y in zip(X,Y))
        Ssigma = sum(1 / y.sd / y.sd for y in Y)
        a = (Sx*Sy - Ssigma*Sxy) / (Sx*Sx - Ssigma*Sxx)
        b = 1/Ssigma * (Sy - a * Sx)
        self.cov = 1 / (Sxx*Ssigma - Sx*Sx) * np.matrix([[Ssigma, -Sx], [-Sx, Sxx]])
        self.value = [Magnitude(a, sqrt(self.cov[0,0]), self.df), Magnitude(b, sqrt(self.cov[1,1]), self.df)]
        self.a = self.value[0]
        self.b = self.value[1]
        self.yval = lambda x: a*x + b
        self.sy = lambda x: sqrt(x*x*self.cov[0,0] + self.cov[1,1] + 2*x*self.cov[0,1])
        self.testchi2 = float(sum((y.val - self.yval(x.val))**2 / y.sd / y.sd for x,y in zip(X,Y)))
        self.pvalue = scipy.stats.chi2.sf(self.testchi2, self.df)


class LinearRegressionNoErrors(Regression):
    r"""
    Class used to hold everything related to a linear regression. It does not take into account any
    data errors.
    Inherits from Regression class. 

    In this case the error of the data is estimated to achieve the best fit, so it
    is used as another parameter of fitting instead of using the data errors.
    The fitting is computed with the (famous) analityc solution of a linear regression that is 
    usually found on many related bibliography. The error
    For more information look up Regression class.


    METHODS:

    For more information look up Regression class.

    MEMBERS:

    - Y, X -- the arrays implied in the regression (of type Magnitude).
    - data -- list of the pairs of Magnitudes (x,y).
    - n -- number of components of Y or X (it must be the same).
    - df --  degrees of freedom of the model.
    - a -- the slope.
    - b -- the y-intercept.
    - value -- maximun likelihood estimators of the model parameters, according to the data.
    - cov -- covariance matrix of the parameters estimators.
    - R2 -- contains the determination coeficient R-square parameter.

    
    INPUT:

    - Y, X -- arrays of magnitudes to fit.

    OUTPUT:

    Returns a Regression type object.
     
    EXAMPLES:

        a = LinearRegressionNoErrors(A, B) 

    """
    def __init__(self, Y, X):
        if len(X) != len(Y):
            raise Exception("Length of X is different from Length of Y")
        self.Y = Y
        self.X = X
        self.data = list(zip([x.val for x in X], [y.val for y in Y]))
        self.n = n = len(X)
        self.df = n - 2
        Sx = sum(x.val for x in X)
        Sy = sum(y.val for y in Y)
        Sxx = sum(x.val**2 for x in X)
        Sxy = sum(x.val*y.val for x,y in zip(X,Y))
        Syy = sum(y.val**2 for y in Y)
        a = (Sx*Sy - n*Sxy) / (Sx*Sx - n*Sxx)
        b = 1/n * (Sy - a * Sx)
        Sepseps = sum((y.val - a*x.val - b)**2 for x,y in zip(X,Y))
        self.cov = 1/n * Sepseps / (n*Sxx - Sx*Sx) * np.matrix([[n, -Sx], [-Sx, Sxx]])
        self.value = [Magnitude(a, sqrt(self.cov[0,0]), self.df), Magnitude(b, sqrt(self.cov[1,1]), self.df)]
        self.a = self.value[0]
        self.b = self.value[1]
        r = (n*Sxy - Sx*Sy) / sqrt((n*Sxx - Sx**2) * (n*Syy - Sy**2))
        self.R2 = r**2
        self.yval = lambda x: a*x + b
        self.sy = lambda x: sqrt(x*x*self.cov[0,0] + self.cov[1,1] + 2*x*self.cov[0,1])

    def graphic(self, pointsSize=40, alpha=twosigma, plotColor="blue", fillAlpha=0.25, errBarFactor = 45, 
        showErrorBars=True, showConfidenceRegion=True, tick_formatter=[cf(),cf()], *args, **kwargs):
        return super().graphic(pointsSize=pointsSize, alpha=alpha, plotColor=plotColor, 
            fillAlpha=fillAlpha, errBarFactor=errBarFactor, showErrorBars=showErrorBars, 
            showConfidenceRegion=showConfidenceRegion, tick_formatter=[cf(),cf()],
            legendPlot=fit_legend(self.a.val, self.b.val, self.R2, showR2=True), *args, **kwargs)


class ErrorPropagation(Magnitude):
    r"""
    Class used to represent an object containing everything related to error propagation
    procedure for a Magnitude. This is, in fact, a Magnitude.

    "All in one" class for grouping some relevant parameters of an error propagation. That
    is to say, the final resulting Magnitude, the value of the cuadratic contributions of
    the formula, the symbolic expresion and value (evaluated partials) of their partials with 
    respect to each parameter, the standard deviation of each parameter, the degrees of freedom of
    each parameter and the result. This class also provides a table output that shows all 
    mentioned above in a more visual form.


    METHODS:

    - table -- function that returns a sage table containing important data related with the error
            propagation procedure.

    MEMBERS:

    - f -- symbolic expresion that sets the dependence of the final variable with the others.
    - vars -- array of variables that f depends on.
    - partials -- array of symbolic partial derivatives of f with respect to each variable.
    - Mvalues -- an array, where each component is the numeric value of each Magnitude involved.
    - Msd -- array containing the standard deviations of each Magnitude variable involved.
    - Mdf -- array with the degrees of freedom of each Magnitude variable involved.
    - contributions -- array holding each cuadratic term of error propagation formula (as components).
    - val -- the final value of the error progation Magnitude.
    - sd -- the final standard deviation of the error progation Magnitude.

    INPUT:

    - f -- symbolic function that gives the error propagation resulting variable.
    - fvars -- array of variables that f depends on.
    - magnitudes -- array of Magnitudes for evaluating the error propagation partials.

    ** It is important to bear in mind that the components of fvars and magnitudes must be 
    supplied in the same exact order.

    OUTPUT:

    Returns a ErrorPropagation type object.
     
    EXAMPLES:

        f(A, B) = A**2 * B/2 #for example
        a = Magnitude(100, 0.1/sqrt(12), 0)
        b = Magnitude(150, 0.5/sqrt(12), 0)
        result = ErrorPropagation(f, [A,B], [a, b]) 
        result --> 7.500·10^5 ± 1.7·10^3 (in latex)
    
    We also can call the object members (listed above). For the individual numbers of the 
    resulting Magnitude:
        result.val --> 750000
        result.sd --> 841.622...
    For other parameters:
        result.f --> symbolic function representing the dependence on the variables of the
                    resulting Magnitude.
        result.vars --> [A,B]
        result.partials --> symbolic functions two component array with partials of f respect
                        to each variable.
        result.Mvalues --> [100,150]
        result.Msd --> [0.0288... , 0.1443...]
        result.Mdf --> [0, 0]
        result.contributions --> [187500, 520833.3...]
        
    
    """
    def __init__(self, f, fvars, magnitudes):
        self.f = f
        self.vars = fvars
        self.partials = [diff(f, X) for X in fvars]
        self.Mvalues = [X.val for X in magnitudes]
        self.Msd = [X.sd for X in magnitudes]
        self.Mdf = [X.df for X in magnitudes]
        # Final degrees of freedom
        try:
            self.df = min(filter(lambda x: x!=0, self.Mdf))
        except:
            self.df = 0
        self.contributions = [X(*self.Mvalues)**2 * Y**2 for X,Y in zip(self.partials, self.Msd)]
        self.val = f(*self.Mvalues)
        self.sd = sqrt(sum(X for X in self.contributions))

    def table(self, pre=3, liminf=-3, limsup=3):
        r"""

        It creates a table with the desired precision, which includes all the information relevant
        to the error propagation. 

        INPUT

        - pre -- the number of significant digits. (default: 3)
        - limsup -- if log10(x) is greater than this parameter, scientific notation will
                    be used. (default: 3)
        - liminf -- if log10(x) is smaller than this parameter, scientific notation will
                    be used. (default: -3)
        
        For more information look up format_number()


        """
        T = table(columns=[self.vars + [self.f(*self.vars)], 
            ["$" + format_number(X, pre=pre, liminf=liminf, limsup=limsup, latex=True) + "$" for X in (self.Mvalues + [self.val])], 
            ["$" + format_number(X, pre=pre, liminf=liminf, limsup=limsup, latex=True) + "$" for X in (self.Msd + [self.sd])], 
            [X if X>0 else r"$\infty$" for X in (self.Mdf + [self.df])], 
            [X(*self.vars) for X in self.partials] + [""],
            ["$" + format_number(X, pre=pre, liminf=liminf, limsup=limsup, latex=True) + "$" for X in self.contributions] + [""]],
            align="center", frame=True, header_row=["Variable", "Valor", r"$s_v$", "gr. lib.", r"$\frac{\partial f}{\partial v}$",
            r"$\left[\frac{\partial f}{\partial v}\right]_{v_{1},\dots,v_{n}}^{2}s_{v}^{2}$"])
        return T


class CombineResults(Magnitude):
    r"""
    Combines results of type Magnitude. This is, in fact, a Magnitude.

    Combines the given Magnitudes with the weighted average formula. This is the
    best estimator for the mean when working with Magnitudes that have the same theoretical
    expected value but different standard deviations. 

    INPUT:

    - *args -- a sequence of Magnitudes.

    OUTPUT:

    The combine result of the given Magnitudes. (A Magnitude)
     
    EXAMPLES:

    If a and b are Magnitudes:
        a = Magnitude(100, 1)
        b = Magnitude(200, 0.5)

        CombineResults(a, b) --> 180.0 ± 0.9 (Resulting magnitude with two sigma confidence, 
                                                as it is set by default for Magnitudes)

    If we have a vector of Magnitudes v:
        CombineResults(*v)

    """
    def __init__(self, *args):
        self.sd = sqrt(1 / sum(1/X.sd**2 for X in args) )
        self.val = sum(X.val/X.sd**2 for X in args) * self.sd**2
        # Final degrees of freedom
        try:
            self.df = min(filter(lambda x: x!=0, [X.df for X in args]))
        except:
            self.df = 0


def polynomicFit(Y, X, n):
    n += 1
    if len(X) != len(Y):
        raise Exception("Length of X is different from Length of Y")
    m = len(X)
    M = matrix(RR, m, n, 0)
    for i in range(0,m):
        for j in range(0,n):
            M[i,j] = X[i]**j
    return np.linalg.solve(M.transpose()*M,M.transpose()*vector(Y))


def format_R2(x):
    r"""
    Returns a represenation of 'x' that ends with the first digit that is not a 9.

    INPUT:

    - x -- a number or a string that represents a number.

    OUTPUT:

    A string that represents the number.

    ALGORITHM:

    We eliminate all ending digits until the end of the string is as follows:
    ...999XY, where X and Y are two numbers different from 9. Now if Y is greater or
    equal to five, we replace X with X+1. Finally we strip the last number.
    """
    if x == 1:
        return "1"
    x = str(float(x))
    j = 2
    while x[j] == "9":
        j += 1
    if int(x[j+1]) >= 5:
        x = x[:j] + str(int(x[j]) + 1)
    else:
        x = x[:j+1]
    return x

def fit_legend(p,c, R2=0, showR2=False):
    """
    Returns a latex string that represents the legend obtained from a LinearFit.

    INPUT:

    - A -- a LinearFit.

    OUTPUT:

    A latex string.

    """
    text = "$" + format_number(p, latex=True) + r"\cdot x"\
        + format_number(c, signed=True, latex=True)
    if showR2:
        text += "\;\;\;R^2=" + format_R2(R2) + "$"
    else:
        text += "$"
    return text


# def grad_descent(g, dg, x, tol=1e-10, Niter = 1000):
#     k = 1
#     x0 = np.array(x)
#     print(g(x0))
#     x1 = x0 - 1e-3*np.array(dg(x0)) / np.linalg.norm(np.array(dg(x0)))
#     while (k < Niter):
#         print(g(x1))
#         vec = np.array(dg(x1)) - np.array(dg(x0))
#         alpha = abs(np.dot(x1 - x0, vec)) / np.dot(vec, vec)
#         x0 = x1
#         x1 = x0 - alpha*np.array(dg(x0))
#         if (np.linalg.norm(x1 - x0) < tol):
#             return x1

def grad_descent(g, dg, x, tol=1e-10, Niter = 1000):
    r"""
    Gradient descent algorithm.

    INPUT:

    - g -- function to find solution of.
    - dg -- derivative of the function g.
    - x --  initial guess of the solution (point to start algorithm).
    - tol -- parameter to fix precision of the solution found.
    - Niter -- maximum number of iterations to let gradient descent to do.

    OUTPUT:

    Hopefully, the solution of the given function (g) with an error fixed by
    input parameter "tol".
    
    EXAMPLES:


    """
    k = 1
    x = np.array(x)
    while(k < Niter):
        print(g(x))
        g1 = g(x)
        z = np.array(dg(x))
        z0 = np.linalg.norm(z)
        if z0==0:
            return x
        z = z/z0
        alpha1 = float(0)
        alpha3 = float(1)
        g3 = g(x - alpha3*z)
        while g3>=g1:
            alpha3 = alpha3/2
            g3 = g(x - alpha3*z)
            if alpha3 < tol/2:
                return x
        alpha2 = alpha3/2
        g2 = g(x - alpha2*z)
        h1 = (g2 - g1) / alpha2
        h2 = (g3 - g2) / (alpha3 - alpha2)
        h3 = (h2 - h1) / alpha3
        alpha0 = 0.5*(alpha2 - h1/h3)
        g0 = g(x - alpha0*z)
        gbar = min(g0, g3)
        alpha = alpha0 if g0 == gbar else alpha3
        x = x - alpha*z
        if abs(gbar - g1) < tol:
            return x
        k = k+1
    raise Exception("Maximum iterations exceeded.")

