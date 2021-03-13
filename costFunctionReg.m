function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = X * theta;
h = sigmoid(h);

for i = 1:m,
	first = -y(i) * log(h(i));
	second = -(1-y(i)) * log(1-h(i));
	J = J + first + second;
end;
J = J / m;

Reg = sum(theta .^ 2) - theta(1) ^ 2;
Reg = Reg * (lambda/(2*m));

J = J + Reg;

grad = 1/m * (X' * (h-y));

for j = 2:n,
	grad(j) = grad(j) + theta(j) * (lambda/m);
end;



% =============================================================

end
