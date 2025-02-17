%% Script Name: 2D channel flow, Finite-Volume, SIMPLE algorithm implementation
% Original script: Tanmay Agrawal, 19.03.2021
% Editor: David Buchner and Angelina Jia
% Last updated: 0.09.2024
% Description:

%% Clear cache
clear all
close all
clc

is_negative = 0; %the factor to det if negative, 0 if value is 0 or positive

%% Defining the problem domain
% Physical domain dimensions
% Note: The critial Re number for Plane Poiseuille Flow is 1000-2000
% make sure this is not exceeded, as this is a laminar solver and will
% otherwise diverge
L_dom = 1; % [m] domain length
H_dom = 0.1; % [m] domain height
rho_phy = 1000; % [kg/m3] physical density of fluid
mu_phy = 0.001; % [kg m-1 s-1] pysical dynamic viscosity of fluid
U_in = 0.01;  % [m/s] inlet velocity

% Reference values for non-dimensionalization
rho_ref = 1000; % [kg/m3] density of water
mu_ref = 0.001; % [kg m-1 s-1] dynamic viscosity of water

Re_ref = rho_ref*U_in*H_dom/mu_ref; % [-] reference Reynolds number

dom_length_x = L_dom/H_dom; %dimensionless length/heights? 
dom_length_y = H_dom/H_dom;
% Spatial discretization, Number of grid cells
n_points_y = 30; 
n_points_x = (dom_length_x/dom_length_y)*(n_points_y); 

top_vel = 0;
bot_vel = 0;
inlet_u=U_in;

if inlet_u <0
    is_negative = 1;
    inlet_u = inlet_u*(-1); %change sign of inlet so converges
    top_vel = top_vel*(-1); %flip top velocity
    bot_vel = bot_vel*(-1); %flip bottom velocity 
end

h = dom_length_y/(n_points_y-1); %grid spacing assumed uniform

% Fluid properties, change with reynolds num
mu = 1/Re_ref; % non-dimensional factor/viscosity
rho = rho_phy/rho_ref; %non dimensionalized density
% Under-relaxation factors
alpha = 0.1;
alpha_p = 0.8;
%depending on non linearity/Re, may need to reduce factors 

%% Initializing the variables
% Final collocated variables / Physical variables
u_final(n_points_y,n_points_x) = 0;
v_final(n_points_y,n_points_x) = 0;
p_final(n_points_y,n_points_x) = 0;
u_final(:,1) = inlet_u;
u_final(:,n_points_x) = inlet_u;

% Staggered variables / Numerical variables
u(n_points_y+1,n_points_x) = 0;
ustar(n_points_y+1,n_points_x) = 0; %intermedite value
d_u(n_points_y+1,n_points_x) = 0;
v(n_points_y,n_points_x+1) = 0;
vstar(n_points_y,n_points_x+1) = 0;
d_v(n_points_y,n_points_x+1) = 0;
p(n_points_y+1,n_points_x+1) = 0;
pstar(n_points_y+1,n_points_x+1) = 0;
pcor(n_points_y+1,n_points_x+1) = 0; %p prime, pressure correction
b(n_points_y+1,n_points_x+1) = 0;
% boundary conditions for channel flow
u(:,1) = inlet_u; 
u(:,n_points_x) = inlet_u;
ustar(:,1) = inlet_u;
ustar(:,n_points_x) = inlet_u;

% Temporary variables for correction steps
u_new(n_points_y+1,n_points_x) = 0;
v_new(n_points_y,n_points_x+1) = 0;
p_new(n_points_y+1,n_points_x+1) = 0;
u_new(:,1) = inlet_u;
u_new(:,n_points_x) = inlet_u;

%% Solving the governing equations
error = 1; % Initializing error
iterations = 0;
error_req = 1e-7; % Definition of convegences criterium/residual
figure(1);

while error > error_req
    % x-momentum eq. interior
    for i = 2:n_points_y
        for j = 2:n_points_x-1
            %coefficients of neighboring grid points
            a_E = (mu/h) - (rho*(u(i,j)+u(i,j+1))/4);
            a_W = (mu/h) + (rho*(u(i,j)+u(i,j-1))/4);
            a_N = (mu/h) - (rho*(v(i-1,j)+v(i-1,j+1))/4);
            a_S = (mu/h) + (rho*(v(i,j)+v(i,j+1))/4);
            %coefficient of grid point in focus
            a_P = a_E + a_W + a_N + a_S + rho*((u(i,j)+u(i,j+1))/2 - (u(i,j)+u(i,j-1))/2 + (v(i-1,j)+v(i-1,j+1))/2 - (v(i,j)+v(i,j+1))/2);
            d_u(i,j) = h/a_P; %later used in continutiy eq
            ustar_mid = (a_E*u(i,j+1) + a_W*u(i,j-1) + a_N*u(i-1,j) + a_S*u(i+1,j) + (p(i,j)-p(i,j+1))*h)/a_P;
            ustar(i,j) = (1-alpha)*u(i,j) + alpha*ustar_mid; %weighted average of u and ustar_mid
        end
    end
    
    %y-momentum eq. interior
    for i = 2:n_points_y-1
        for j = 2:n_points_x
            a_E = (mu/h) - (rho*(u(i,j)+u(i+1,j))/4);
            a_W = (mu/h) + (rho*(u(i,j-1)+u(i+1,j-1))/4);
            a_N = (mu/h) - (rho*(v(i,j)+v(i-1,j))/4);
            a_S = (mu/h) + (rho*(v(i,j)+v(i+1,j))/4);
            a_P = a_E + a_W + a_N + a_S + rho*((u(i,j)+u(i+1,j))/2 - (u(i,j-1)+u(i+1,j-1))/2 + (v(i,j)+v(i-1,j))/2 - (v(i,j)+v(i+1,j))/2);
            d_v(i,j) = h/a_P;
            vstar_mid = (a_E*v(i,j+1) + a_W*v(i,j-1) + a_N*v(i-1,j) + a_S*v(i+1,j) + (p(i+1,j)-p(i,j))*h)/a_P;
            vstar(i,j) = (1-alpha)*v(i,j) + alpha*vstar_mid; %same change, weigthed average
        end
    end
    
    % x-momentum eq. boundary
    ustar(:,1) = inlet_u; % left wall BC, constant velocity
    ustar(:,n_points_x) = ustar(:,n_points_x-1);% right wall BC, zero velocity gradient
    ustar(1,2:n_points_x) = 2*top_vel-ustar(2,2:n_points_x); % top wall BC, no slip
    ustar(n_points_y+1,2:n_points_x) = 2*bot_vel-ustar(n_points_y,2:n_points_x); % bottom wall BC
    %boundary lies between the u vel points, so the average = the wall
    %velocity, if exactly opposite will =0. 

    % y-momentum eq. boundary
    vstar(:,1) = vstar(:,2); % left wall BC
    vstar(:,n_points_x+1) = vstar(:,n_points_x); % right wall BC
    vstar(1,2:n_points_x) = 0; % top wall BC
    vstar(n_points_y,2:n_points_x) = 0; % bottom wall B
    %v-vel lies on the boundaries, so no averaging needed
    %top and bot wall will always be zero, since we assume the plates are
    %horizontal


    % Zeroing the corrections to begin with
    pcor(1:n_points_y+1,1:n_points_x+1) = 0;
    
    % Continuity eq. aka pressure correction - interior
    for i = 2:n_points_y
        for j = 2:n_points_x
            a_E = rho*h*d_u(i,j);
            a_W = rho*h*d_u(i,j-1);
            a_N = rho*h*d_v(i-1,j);
            a_S = rho*h*d_v(i,j);
            a_P = a_E + a_W + a_N + a_S;
            b(i,j) = rho*(h*(ustar(i,j-1)-ustar(i,j)) + h*(vstar(i,j)-vstar(i-1,j)));           
            pcor(i,j) = (a_E*pcor(i,j+1) + a_W*pcor(i,j-1) + a_N*pcor(i-1,j) + a_S*pcor(i+1,j) + b(i,j))/a_P;
        end
    end
    
    % Correcting the pressure field
    for i = 2:n_points_y
        for j = 2:n_points_x
            p_new(i,j) = p(i,j) + alpha_p*pcor(i,j);
        end
    end
    
    % Correcting the velocities, alpha terms already applied for velocities
    for i = 2:n_points_y
        for j = 2:n_points_x-1
            u_new(i,j) = ustar(i,j) + d_u(i,j)*(pcor(i,j)-pcor(i,j+1));
        end
    end
    
    for i = 2:n_points_y-1
        for j = 2:n_points_x
            v_new(i,j) = vstar(i,j) + d_v(i,j)*(pcor(i+1,j)-pcor(i,j));
        end
    end
    
   %reimplement boundary conditions after each correction 
    % Continuity eq boundary
    p_new(1:n_points_y,1) = p_new(1:n_points_y,2); % left wall BC, uniform pressure inlet
    p_new(1:n_points_y,n_points_x+1) = p_new(1:n_points_y,n_points_x); % right wall BC, pressure outlet
    p_new(1,1:n_points_x) = p_new(2,1:n_points_x); % top wall BC
    p_new(n_points_y+1,1:n_points_x) = p_new(n_points_y,1:n_points_x); % bottom wall BC

    % x-momentum eq. boundary
    u_new(:,1) = inlet_u; % left wall BC, constant velocity
    u_new(:,n_points_x) = u_new(:,n_points_x-1);% right wall BC, zero velocity gradient
    u_new(1,2:n_points_x) = 2*top_vel-u_new(2,2:n_points_x); % top wall BC
    u_new(n_points_y+1,2:n_points_x) = 2*bot_vel-u_new(n_points_y,2:n_points_x); % bottom wall BC

    % y-momentum eq. boundary
    v_new(2:n_points_y-1,1) = v_new(2:n_points_y-1,2); % left wall BC
    v_new(2:n_points_y-1,n_points_x+1) = v_new(2:n_points_y-1,n_points_x); % right wall BC
    v_new(1,:) = 0; % top wall BC
    v_new(n_points_y,:) = 0; % bottom wall BC
    
    % continuity eq. residual as error measure
    error = 0;
    for i = 2:n_points_y
        for j = 2:n_points_x
            error = error + abs(b(i,j));
        end
    end
    
    % error monitoring after every few timesteps
    if(rem(iterations,1000)) == 0
        figure(1);
        semilogy(iterations, error, '-ko')
        hold on
        xlabel('Iterations');
        ylabel('Residual Error');
    end
    
    % Finishing the iteration
    u = u_new;
    v = v_new;
    p = p_new;
    iterations = iterations + 1;
end

% Mapping the staggered variables to collocated variables (all back onto
% one grid)
for i = 1:n_points_y
    for j = 1:n_points_x
        u_final(i,j) = 0.5*(u(i,j) + u(i+1,j));
        v_final(i,j) = 0.5*(v(i,j) + v(i,j+1));
        p_final(i,j) = 0.25*(p(i,j) + p(i,j+1) + p(i+1,j) + p(i+1,j+1));
    end
end

%% Contour and vector visuals.
%{
%before edits
            x_dom = ((1:n_points_x)-1).*h;
            y_dom = 1-((1:n_points_y)-1).*h;
            [X,Y] = meshgrid(x_dom,y_dom);
%}

            x_dom = (((1:n_points_x)-1).*h).*H_dom;
            y_dom = (1-((1:n_points_y)-1).*h).*H_dom;
            [X,Y] = meshgrid(x_dom,y_dom);

%** is_negative =1, need to invert the x-axis 
if is_negative ==1
    u_final = -1*flip(u_final,2); %columns should be flipped, but rows remain
    v_final = flip(v_final,2);
    p_final = flip(p_final,2);
end
u_average = sum(u_final(:,n_points_x)) / n_points_y;
G_term  = ((((top_vel-bot_vel)/2)+bot_vel-u_average)*12*mu_phy)/(H_dom^2);
%u-velocity contour
figure;
contourf(X,Y, u_final, 21, 'LineStyle', 'none')
colorbar
colormap('jet')
xlabel('x')
ylabel('y')
title('U- Velocity')
hold on
streamslice(X, Y, u_final, v_final, 'noarrows','k');
%daspect([1 0.5 2]) %**plot axis fix the scale here 
pbaspect([1 1 1])
hold off; 

%v-velocity contour
figure;
contourf(X,Y, v_final, 21, 'LineStyle', 'none')
colorbar
colormap('jet')
xlabel('x')
ylabel('y')
title('V- Velocity')
pbaspect([1.2 1 1])
hold on
streamslice(X, Y, u_final, v_final,  0.5,'noarrows','k');
%daspect([1 0.5 2])
hold off; 

%overall velocity contour
vel_mag = sqrt(u_final.^2+v_final.^2);
figure;
contourf(X,Y, vel_mag, 21, 'LineStyle', 'none')
colorbar
colormap('jet')
xlabel('x')
ylabel('y')
title('Overall Velocity')
pbaspect([1 1 1])
hold on
streamslice(X, Y, u_final, v_final, 'noarrows','k');
hold off; 

%pressure contour
figure;
contourf(X,Y, p_final, 21, 'LineStyle', 'none')
colorbar
colormap('jet')
xlabel('x')
ylabel('y')
title('Pressure')
pbaspect([1 1 1])


%quiver plot for velocities
figure;
hold on
quiver(X, Y, u_final, v_final, 1, 'k')
pbaspect([1 1 1])


%% Plotting the profile evolution
%velocity profile in the x-direction, to show the profiles sketched in class

% set up 
y_lin= linspace(0,dom_length_y,n_points_y)*H_dom;
y_lin = flip(y_lin); 
x_lin= linspace(0,dom_length_x,n_points_x)*H_dom;
max_vel = max(u_final(:,n_points_x)); 
min_vel = min(u_final(:,n_points_x));
%** adjust values if inlet is on oppostie side
if is_negative ==1
    max_vel = max(u_final(:,1)); 
    min_vel = min(u_final(:,1));
end 


%graphs, can change x_target to see different points along the length of
%the channel 
x_target =1; %location in meters, in the x-direction
[~, xIndex] = min(abs(x_lin - x_target));
Vel = u_final(:, xIndex);
figure;
plot(Vel, y_lin, '-');
xlim([min_vel-.001,max_vel]);
ylim([0,H_dom]);
title('velocity profile at x = ', num2str(x_target));
grid on;

x_target =0.02;
[~, xIndex] = min(abs(x_lin - x_target));
Vel = u_final(:, xIndex);
figure;
plot(Vel, y_lin, '-');
xlim([min_vel-.001,max_vel]);
ylim([0,H_dom]);
title('velocity profile at x = ', num2str(x_target));
grid on;

x_target = 0.005;  
[~, xIndex] = min(abs(x_lin - x_target));
Vel = u_final(:, xIndex);
figure;
plot(Vel, y_lin, '-');
xlim([min_vel-.001,max_vel]);
ylim([0,H_dom]);
title('velocity profile at x = ', num2str(x_target));
grid on;

x_target =0;
[~, xIndex] = min(abs(x_lin - x_target));
Vel = u_final(:, xIndex);
figure;
plot(Vel, y_lin, '-');
xlim([min_vel-.001,max_vel]);
ylim([0,H_dom]);
title('velocity profile at x = ', num2str(x_target));
grid on;
