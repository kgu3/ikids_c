// The parameters returned by fitting an ellipse onto the target
struct ellipsePara {
	double alpha_est;	// in radian
	double gama_est;
	double EllipsePara[5]; // EllipsePara = EllipsePara=[zz[2],aa,bb,alpha]
	int shape; // -1 = 'none', 0 = 'fat', 1 = 'tall'
};