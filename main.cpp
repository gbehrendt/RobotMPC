#define _USE_MATH_DEFINES

#include <iostream>
#include <casadi/casadi.hpp>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <fstream>
#include </usr/local/include/eigen3/unsupported/Eigen/CXX11/Tensor>
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

using namespace Eigen;
using namespace casadi;
using namespace std;

void shift(int, double, MatrixXd &, MatrixXd, MatrixXd &);

int main() {

    // Declare states + controls
    SX v = SX::sym("v");
    SX omega = SX::sym("omega");
    SX u = vertcat(v,omega);

    SX xPos = SX::sym("xPos");
    SX yPos = SX::sym("yPos");
    SX theta = SX::sym("theta");
    SX x = vertcat(xPos,yPos,theta);

    // Number of differential states
    int numStates = x.size1();

    // Number of controls
    int numControls = u.size1();

    // Bounds and initial guess for the control
    double vMax = 0.6;
    double vMin = -vMax;
    double omegaMax = M_PI/4;
    double omegaMin = -omegaMax;

    // Bounds and initial guess for the control
    std::vector<double> u_min =  { vMin, omegaMin };
    std::vector<double> u_max  = { vMax, omegaMax  };
    std::vector<double> u_init = {  0.0, 0.0  };

    // Bounds and initial guess for the state
    std::vector<double> x0_min = {   0,    0,   0 };
    std::vector<double> x0_max = {   0,    0,   0 };
    std::vector<double> x_min  = {-2, -2, -inf };
    std::vector<double> x_max  = {2, 2, inf };
    std::vector<double> xf_min = {   1.5,    1.5,    0.0 };
    std::vector<double> xf_max = {   1.5,    1.5,    0.0 };
    std::vector<double> x_init = {   0,    0,   0 };

    // Final time
    // int N = 10; // Prediction Horizon
    int N = 50; // Number of shooting nodes
    double ts = 0.2; // sampling period
    double tf = ts*N;

    // ODE right hand side and quadrature
    SX ode = vertcat(v*cos(theta), v*sin(theta), omega);
    SX quad = v*v + theta*theta + xPos*xPos + yPos*yPos + omega*omega;
    SXDict dae = {{"x", x}, {"p", u}, {"ode", ode}, {"quad", quad}};
    SXDict dae2 = {{"x", x}, {"p", u}, {"ode", ode}};

    // Create an integrator (CVodes)
    //Function F = integrator("integrator", "cvodes", dae, 0, tf/N);
    Function F = integrator("integrator", "rk", dae2, 0, tf/N);

    // Total number of NLP variables
    const int numVars = numStates*(N+1) + numControls*N;

    // Declare variable vector for the NLP
    MX V = MX::sym("V",numVars);

    // NLP variable bounds and initial guess
    std::vector<double> v_min,v_max,v_init;

    // Offset in V
    int offset=0;

    // State at each shooting node and control for each shooting interval
    std::vector<MX> X, U;
    for(int k=0; k<N; ++k){
        // Local state
        X.push_back( V.nz(Slice(offset,offset + numStates)));
        if(k==0){
            v_min.insert(v_min.end(), x0_min.begin(), x0_min.end());
            v_max.insert(v_max.end(), x0_max.begin(), x0_max.end());
        } else {
            v_min.insert(v_min.end(), x_min.begin(), x_min.end());
            v_max.insert(v_max.end(), x_max.begin(), x_max.end());
        }
        v_init.insert(v_init.end(), x_init.begin(), x_init.end());
        offset += numStates;

        // Local control
        U.push_back( V.nz(Slice(offset,offset + numControls)));
        v_min.insert(v_min.end(), u_min.begin(), u_min.end());
        v_max.insert(v_max.end(), u_max.begin(), u_max.end());
        v_init.insert(v_init.end(), u_init.begin(), u_init.end());
        offset += numControls;
    }

    // State at end
    X.push_back(V.nz(Slice(offset,offset+numStates)));
    v_min.insert(v_min.end(), xf_min.begin(), xf_min.end());
    v_max.insert(v_max.end(), xf_max.begin(), xf_max.end());
    v_init.insert(v_init.end(), x_init.begin(), x_init.end());
    offset += numStates;

//    cout << "v_min:" << endl << v_min << endl;
//    cout << "v_max:" << endl << v_max << endl;

    // Make sure that the size of the variable vector is consistent with the number of variables that we have referenced
    casadi_assert(offset==numVars, "");

    // Initialize Objective Function and Weighting Matrices
    MX J = 0;
    MX Q = MX::zeros(numStates,numStates);
    Q(0,0) = 1;
    Q(1,1) = 5;
    Q(2,2) = 0.05;

    MX R = MX::zeros(numControls,numControls);
    R(0,0) = 0.5;
    R(1,1) = 0.05;

    MX xd = MX::zeros(numStates);
    xd(0) = 1.5;
    xd(1) = 1.5;
    xd(2) = 0.0;

    //Constraint function and bounds
    std::vector<MX> g;

    // Loop over shooting nodes
    for(int k=0; k<N; ++k){
        // Create an evaluation node
        MXDict I_out = F(MXDict{{"x0", X[k]}, {"p", U[k]}});

        // Save continuity constraints
        g.push_back( I_out.at("xf") - X[k+1] );

        // Add objective function contribution
        J += mtimes(mtimes((I_out.at("xf")-xd).T(),Q),(I_out.at("xf")-xd)) + mtimes(mtimes(U[k].T(),R),U[k]);
    }

    // NLP
    MXDict nlp = {{"x", V}, {"f", J}, {"g", vertcat(g)}};

    // Set options
    Dict opts;
    opts["ipopt.tol"] = 1e-5;
    opts["ipopt.max_iter"] = 5;
    opts["ipopt.print_level"] = 0;
    opts["ipopt.acceptable_tol"] = 1e-8;
    opts["ipopt.acceptable_obj_change_tol"] = 1e-6;
    opts["ipopt.file_print_level"] = 3;
    opts["ipopt.print_timing_statistics"] = "yes";
    opts["ipopt.output_file"] = "timing.csv";

    // Create an NLP solver and buffers
    Function solver = nlpsol("nlpsol", "ipopt", nlp, opts);
    std::map<std::string, DM> arg, res, sol;

    // Bounds and initial guess
    arg["lbx"] = v_min;
    arg["ubx"] = v_max;
    arg["lbg"] = 0;
    arg["ubg"] = 0;
    arg["x0"] = v_init;
    //arg["p"] = 0;



    //---------------------//
    //      MPC Loop       //
    //---------------------//
    Eigen::MatrixXd x0(3,1);
    x0 << 0.0, 0.0, 0.0;
    Eigen::MatrixXd xs(3,1);
    xs << 1.5,1.5,0;

    Eigen::MatrixXd xx(numStates, N+1);
    xx.col(0) = x0;

    Eigen::MatrixXd xx1(numStates, N+1);
    Eigen::MatrixXd X0(numStates,N+1);
    Eigen::MatrixXd u0;
    Eigen::MatrixXd uwu(numControls,N);
    Eigen::MatrixXd u_cl(numControls,N);

    vector<vector<double> > MPCstates(numStates);
    vector<vector<double> > MPCcontrols(numControls);

    // Start MPC
    int iter = 0;
    double tol = 1e-2;
    double epsilon = 1e-2;
    Eigen::VectorXf t(N);
    t(0) = 0;
    double infNorm = 100;

    while(infNorm > epsilon && iter <= N)
    {
        // Solve NLP
        sol = solver(arg);

        std::vector<double> V_opt(sol.at("x"));
        Eigen::MatrixXd V = Eigen::Map<Eigen::Matrix<double, 503, 1> >(V_opt.data());

        // Store Solution
        for(int i=0; i<=N; ++i)
        {
            xx1(0,i) = V(i*(numStates+numControls));
            xx1(1,i) = V(1+i*(numStates+numControls));
            xx1(2,i) = V(2+i*(numStates+numControls));
            if(i < N)
            {
                uwu(0,i)= V(numStates + i*(numStates+numControls));
                uwu(1,i) = V(1+numStates + i*(numStates+numControls));
            }
        }
        cout << "NLP States:" << endl << xx1 << endl;
        cout <<endl;
        cout << "NLP Controls:" << endl <<  uwu << endl;
        cout <<endl;

        // Get solution Trajectory
        u_cl.col(iter) = uwu.col(0); // Store first control action from optimal sequence
        t(iter+1) = t(iter) + ts;

        // Apply control and shift solution
        shift(N,ts,x0,uwu,u0);
        xx(Eigen::placeholders::all,iter+1)=x0;

        // Shift trajectory to initialize next step
        std::vector<int> ind(N) ; // vector with N-1 integers to be filled
        std::iota (std::begin(ind), std::end(ind), 1); // fill vector with N integers starting at 1
        X0 = xx1(Eigen::placeholders::all,ind); // assign X0 with columns 1-(N) of xx1
        X0.conservativeResize(X0.rows(), X0.cols()+1);
        X0.col(X0.cols()-1) = xx1(Eigen::placeholders::all,Eigen::placeholders::last);

        cout << "MPC States:" << endl << xx << endl;
        cout <<endl;
        cout << "MPC Controls:" << endl << u_cl << endl;

        for(int j=0; j<numStates; j++)
        {
            MPCstates[j].push_back(x0(j));
        }

        for(int j=0; j<numControls; j++)
        {
            MPCcontrols[j].push_back(u_cl(j,iter));
        }

        // Re-initialize Problem Parameters
        v_min.erase(v_min.begin(),v_min.begin()+3);
        v_min.insert(v_min.begin(),x0(2));
        v_min.insert(v_min.begin(),x0(1));
        v_min.insert(v_min.begin(),x0(0));

        v_max.erase(v_max.begin(),v_max.begin()+3);
        v_max.insert(v_max.begin(),x0(2));
        v_max.insert(v_max.begin(),x0(1));
        v_max.insert(v_max.begin(),x0(0));

        arg["lbx"] = v_min;
        arg["ubx"] = v_max;
        arg["x0"] = V_opt;
//        arg["x0"] = x00;

        infNorm = max((x0-xs).lpNorm<Eigen::Infinity>(),u_cl.col(iter).lpNorm<Eigen::Infinity>()); // l-infinity norm of current state and control
        cout << infNorm << endl;

        iter++;
        cout << iter <<endl;
    }

    ofstream fout; // declare fout variable
    fout.open("MPCresults.csv", std::ofstream::out | std::ofstream::trunc ); // open file to write to
    fout << "MPC States:" << endl;
    for(int i=0; i < MPCstates.size(); i++)
    {
        fout << MPCstates[i] << endl;
    }
    fout << "MPC Controls:" << endl;
    for(int i=0; i < MPCcontrols.size(); i++)
    {
        fout << MPCcontrols[i] << endl;
    }
    fout.close();

//    plt::figure();
//    plt::plot(MPCstates[0],MPCstates[1]);
//
//    plt::figure();
//    plt::plot(MPCstates[2]);
//
//    plt::figure();
//    plt::plot(MPCcontrols[0]);
//    plt::plot(MPCcontrols[1]);
//
//    plt::show();

    return 0;
}

//////////////////////////////////////////////////////////////////////////////
// Function Name: f
// Description: This function is used to implement the dynamics of our system
// Inputs: MatrixXd st - current state, MatrixXd con - current control action
// Outputs: MatrixXd xDot - time derivative of the current state
//////////////////////////////////////////////////////////////////////////////
MatrixXd f(MatrixXd st, MatrixXd con)
{
    double x = st(0);
    double y = st(1);
    double theta = st(2);
    double v = con(0);
    double omega = con(1);

    MatrixXd xDot(3,1);
    xDot << v*cos(theta), v*sin(theta), omega;
    return xDot;
}

//////////////////////////////////////////////////////////////////////////////
// Function Name: f
// Description: This function is used to shift our MPC states and control inputs
//              in time so that we can re-initialize our optimization problem
//              with the new current state of the system
// Inputs: N - Prediction Horizon, ts - sampling time, x0 - initial state,
//             uwu - optimal control sequence from NLP, u0 - shifted
//             control sequence
// Outputs: None
//////////////////////////////////////////////////////////////////////////////
void shift(int N, double ts, MatrixXd& x0, MatrixXd uwu, MatrixXd& u0)
{
    // Shift State
    MatrixXd st = x0;
    MatrixXd con = uwu.col(0);

    MatrixXd k1 = f(st,con);
    MatrixXd k2 = f(st + (ts/2)*k1,con);
    MatrixXd k3 = f(st + (ts/2)*k2,con);
    MatrixXd k4 = f(st + ts*k3,con);

    st = st + ts/6*(k1 + 2*k2 + 2*k3 + k4);
    x0 = st;

    // Shift Control
    std::vector<int> ind(N-1) ; // vector with N-1 integers to be filled
    std::iota (std::begin(ind), std::end(ind), 1); // fill vector with N-1 integers starting at 1
    u0 = uwu(Eigen::placeholders::all,ind); // assign u0 with columns 1-(N-1) of uwu
    u0.conservativeResize(u0.rows(), u0.cols()+1);
    u0.col(u0.cols()-1) = uwu(Eigen::placeholders::all,Eigen::placeholders::last); // copy last column and append it
}














///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//    // Declare variables
//    SX u = SX::sym("u"); // control
//    SX r = SX::sym("r"), s = SX::sym("s"); // states
//    SX x = vertcat(r,s);
//
//
//    // Number of differential states
//    int nx = x.size1();
//
//    // Number of controls
//    int nu = u.size1();
//
//    // Bounds and initial guess for the control
//    std::vector<double> u_min =  { -0.75 };
//    std::vector<double> u_max  = {  1.0  };
//    std::vector<double> u_init = {  0.0  };
//
//    // Bounds and initial guess for the state
//    std::vector<double> x0_min = {   0,    1 };
//    std::vector<double> x0_max = {   0,    1 };
//    std::vector<double> x_min  = {-inf, -inf };
//    std::vector<double> x_max  = { inf,  inf };
//    std::vector<double> xf_min = {   0,    0 };
//    std::vector<double> xf_max = {   0,    0 };
//    std::vector<double> x_init = {   0,    0 };
//
//    // Final time
//    double tf = 20.0;
//
//    // Number of shooting nodes
//    int ns = 50;
//
//    // ODE right hand side and quadrature
//    SX ode = vertcat((1 - s*s)*r - s + u, r);
//    SX quad = r*r + s*s + u*u;
//    SXDict dae = {{"x", x}, {"p", u}, {"ode", ode}, {"quad", quad}};
//    SXDict dae2 = {{"x", x}, {"p", u}, {"ode", ode}};
//
//    // Create an integrator (CVodes)
//    Function F = integrator("integrator", "cvodes", dae, 0, tf/ns);
//    //Function F = integrator("integrator", "rk", dae2, 0, tf/ns);
//
//    // Total number of NLP variables
//    int NV = nx*(ns+1) + nu*ns;
//
//    // Declare variable vector for the NLP
//    MX V = MX::sym("V",NV);
//
//    // NLP variable bounds and initial guess
//    std::vector<double> v_min,v_max,v_init;
//
//    // Offset in V
//    int offset=0;
//
//    // State at each shooting node and control for each shooting interval
//    std::vector<MX> X, U;
//    for(int k=0; k<ns; ++k){
//        // Local state
//        X.push_back( V.nz(Slice(offset,offset+nx)));
//        if(k==0){
//            v_min.insert(v_min.end(), x0_min.begin(), x0_min.end());
//            v_max.insert(v_max.end(), x0_max.begin(), x0_max.end());
//        } else {
//            v_min.insert(v_min.end(), x_min.begin(), x_min.end());
//            v_max.insert(v_max.end(), x_max.begin(), x_max.end());
//        }
//        v_init.insert(v_init.end(), x_init.begin(), x_init.end());
//        offset += nx;
//
//        // Local control
//        U.push_back( V.nz(Slice(offset,offset+nu)));
//        v_min.insert(v_min.end(), u_min.begin(), u_min.end());
//        v_max.insert(v_max.end(), u_max.begin(), u_max.end());
//        v_init.insert(v_init.end(), u_init.begin(), u_init.end());
//        offset += nu;
//    }
//
//    // State at end
//    X.push_back(V.nz(Slice(offset,offset+nx)));
//    v_min.insert(v_min.end(), xf_min.begin(), xf_min.end());
//    v_max.insert(v_max.end(), xf_max.begin(), xf_max.end());
//    v_init.insert(v_init.end(), x_init.begin(), x_init.end());
//    offset += nx;
//
//    // Make sure that the size of the variable vector is consistent with the number of variables that we have referenced
//    casadi_assert(offset==NV, "");
//
//    // Objective function
//    MX J = 0;
//    MX obj = 0;
//
//    //Constraint function and bounds
//    std::vector<MX> g;
//
//    // Loop over shooting nodes
//    for(int k=0; k<ns; ++k){
//        // Create an evaluation node
//        MXDict I_out = F(MXDict{{"x0", X[k]}, {"p", U[k]}});
//
//        // Save continuity constraints
//        g.push_back( I_out.at("xf") - X[k+1] );
//        //cout << I_out.at("xf") << endl << X[k] << endl;
//
//        // Add objective function contribution
//        J += I_out.at("qf");
//        //cout << I_out.at("xf") << endl;
//        obj += dot(I_out.at("xf"),I_out.at("xf")) + dot(U[k],U[k]);    //mtimes(mtimes((st - xd).T(),Q),(st - xd)) + mtimes(mtimes(con.T(),R),con);
//    }
//
//    // NLP
//    MXDict nlp = {{"x", V}, {"f", J}, {"g", vertcat(g)}};
//
//    // Set options
//    Dict opts;
//    opts["ipopt.tol"] = 1e-5;
//    opts["ipopt.max_iter"] = 100;
//
//    // Create an NLP solver and buffers
//    Function solver = nlpsol("nlpsol", "ipopt", nlp, opts);
//    std::map<std::string, DM> arg, res;
//
//    // Bounds and initial guess
//    arg["lbx"] = v_min;
//    arg["ubx"] = v_max;
//    arg["lbg"] = 0;
//    arg["ubg"] = 0;
//    arg["x0"] = v_init;
//
//    // Solve the problem
//    res = solver(arg);
//
//    // Optimal solution of the NLP
//    std::vector<double> V_opt(res.at("x"));
//
//    // Get the optimal state trajectory
//    std::vector<double> r_opt(ns+1), s_opt(ns+1);
//    for(int i=0; i<=ns; ++i){
//        r_opt[i] = V_opt.at(i*(nx+1));
//        s_opt[i] = V_opt.at(1+i*(nx+1));
//    }
//    std::cout << "r_opt = " << std::endl << r_opt << std::endl;
//    std::cout << "s_opt = " << std::endl << s_opt << std::endl;
//
//    // Get the optimal control
//    std::vector<double> u_opt(ns);
//    for(int i=0; i<ns; ++i){
//        u_opt[i] = V_opt.at(nx + i*(nx+1));
//    }
//    std::cout << "u_opt = " << std::endl << u_opt << std::endl;
//
//    plt::figure(1);
//    plt::plot(u_opt);
//
//    plt::figure(2);
//    plt::plot(r_opt);
//    plt::plot(s_opt);
//
//    plt::figure(3);
//    plt::plot(V_opt);
//    plt::show();

/*
float T = 0.2;
int N = 10;

// Define input limits
float vMax = 0.6;
float vMin = -vMax;
float omegaMax = M_PI/4;
float omegaMin = -omegaMax;

//vector<double> vMax = {};


// Define states
SX x = SX::sym("x",1);
SX y = SX::sym("y",1);
SX theta = SX::sym("theta",1);
SX states = SX::vertcat({x, y, theta});
int numStates = states->size();
SX Q = SX::zeros(numStates,numStates);
Q(0,0) = 1;
Q(1,1) = 5;
Q(2,2) = 0.05;

// Declare Control Inputs
SX v = SX::sym("v",1);
SX omega = SX::sym("omega",1);
SX controls = SX::vertcat({v, omega});
int numControls = controls->size();

SX R = SX::zeros(numControls,numControls);
R(0,0) = 0.5;
R(1,1) = 0.05;

// Declare Problem Structures
//SX rhs = SX::vertcat({v*cos(theta),v*sin(theta), omega}); // xDot equations
//Function f = Function("f",{states,controls},{rhs}); // nonlinear mapping function xDot = f(x,u)
int numVars = numStates*(N+1) + numControls*N;
int offset=0;
SX Vars = SX::sym("V",numVars);
SX U = SX::sym("U",numControls,N);
SX P = SX::sym("P", numStates*2);
SX X = SX::sym("X",numStates, N+1);
SX obj = 0;
SX g = {};

//////////////////// Construct Constraints ////////////////////
// Initial State Constraint
g = vertcat(g,X(Slice(0,numStates,1),0) - P(Slice(0,numStates,1),0));
//cout << g << endl;

// Build Objective and Construct Continuity Constraints for Multiple Shooting
SX st(numStates,1);
SX stNext(numStates,1);
SX con(numControls,1);
SX xd(numStates,1);
SX k1,k2,k3,k4,stNextRK4;

for(int k=0; k<N; k++)
{
    st = X(Slice(0, numStates,1),k);
    con = U(Slice(0, numControls,1),k);
    xd = P(Slice(numStates,2*numStates,1),0);

    obj = obj + mtimes(mtimes((st - xd).T(),Q),(st - xd)) + mtimes(mtimes(con.T(),R),con);
    stNext = X(Slice(0, numStates,1),k+1);

    // 4th Order Runge Kutta
    k1 = f(st,con);
    k2 = f(st + T/2*k1,con);
    k3 = f(st + T/2*k2,con);
    k4 = f(st + T*k3,con);
    stNextRK4 = st + T/6*(k1 + 2*k2 + 2*k3 + k4);

    g = vertcat(g,stNext - stNextRK4);

    //cout << obj << endl;
    //cout << mtimes(mtimes(st.T(),Q),st) << endl;
    //cout << st.T()*Q << endl;
    //cout << X(Slice(ind1, ind2,1),k) << endl;

}

// Do I have to reshape X and U into vector for nlp dictionary?

// Assemble optimization problem using defined structures
SXDict nlp = {{"x",Vars}, {"f", obj},{"g",g}, {"p",P}};

// Set options
Dict opts;
opts["ipopt.tol"] = 1e-5;
opts["ipopt.max_iter"] = 100;
opts["ipopt.print_level"] = 0;
opts["ipopt.print_time"] = 0;
opts["ipopt.acceptable_tol"] = 1e-8;
opts["ipopt.acceptable_obj_change_tol"] = 1e-6;

// Create an NLP solver and buffers
Function solver = nlpsol("nlpsol", "ipopt", nlp, opts);
std::map<std::string, DM> arg, res;

*/