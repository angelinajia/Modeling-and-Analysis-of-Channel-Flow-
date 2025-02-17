%% Script Name: 1D channel flow, analytical solutions
% Author: Angelina Jia
% Based on CENG50002 - Transder Processes 2 2023-2024 Fluid Mechanics 2 lecture series delivered by Prof. Ronny Pini

% assumptions: steady state, fully dev, unidirectional, dv/dy=0 from continuity
%equation, neglected gravity(flat plate), z-direction not changing 
%analytical solutions for fully developed flow between parallel plates
 function u = combinedFlow(mu, H, n, G, u1, u2)
    % mu: dynamic viscosity (kg/(ms))
    % H: distance between plates (m)
    % n: number of grid points
    % G: pressure gradient in the x-direction (Pa/m)
    % u1: base plate velocity (m/s)
    % u2: upper plate velocity (m/s)
    

    del_y= H/(n-1); %increments of y
    % mu: dynamic viscosity (kg/(ms))
    % H: distance between plates (m)
    % n: number of grid points
    % G: pressure gradient in the x-direction (Pa/m)
    % u1: base plate velocity (m/s)
    % u2: upper plate velocity (m/s)
    
    del_y = H/(n-1); % increments of y
    y = linspace(0, H, n)'; % y discretized into n points
    dist = y;
    
    % Analytical solution for velocity profile
    u = (G/(2*mu)) * (y.^2 - H*y) + ((u2 - u1)/H) * y + u1;
    
    % Analytical solution for shear stress profile
    % Shear stress tau_yx = mu * du/dy
    % Analytical derivative of u(y):
    du_dy = (G/(2*mu)) * (2*y - H) + (u2 - u1)/H;
    tau_yx = mu * du_dy;

    % Plotting the velocity profile
    figure;
    plot(u, dist, '-');
    xlabel('Velocity of Flowing Fluid');
    ylabel('Distance between the plates');
    title('Velocity Profile of Flowing Fluid between Two Parallel Plates');
    grid on;
    hold on;
    yline(H/2, '--');
    xline(0, '--');
    hold off;

    % Plotting the shear stress profile
    figure;
    plot(tau_yx, dist, '-');
    xlabel('Shear Stress of Flowing Fluid');
    ylabel('Distance between the plates');
    title('Stress Profile of Flowing Fluid between Two Parallel Plates');
    grid on;
    hold on;
    yline(H/2, '--');
    xline(0,'--');
    hold off;

    % Find max/min value and its index
    [max_u, index] = max(u);
    [min_u, minIndex] = min(u);

    % y-coordinate corresponding to the max/min value
    y_max = dist(index);
    y_min = dist(minIndex);
    
    fprintf('Max velocity: %.5f m/s \n', max_u);
    fprintf('Height of max velocity: %.2f m \n', y_max);

    fprintf('Min velocity: %.2f m/s \n', min_u);
    fprintf('Height of min velocity: %.2f m \n', y_min);
      
    % Define the length of the plates in the x-direction, since 2D for
    % contour plot 
    L = 1; % assuming a length of 1 meter
    x = linspace(0, L, n); % x-coordinates
    [X, Y] = meshgrid(x, y); % creating a mesh grid with x and y dimensions

    % Replicate the velocity profile across the x-direction
    U = repmat(u, 1, n)'; % Replicates u as rows of 2D matrix, ensure proper orientation with '

    % Plotting the velocity contour
    figure;
    contourf(Y, X, U, 20); % 20 contour levels
    colorbar;
    xlabel('Length of Plates (x)');
    ylabel('Height between Plates (y)');
    title('Velocity Contour Plot');
    grid on;

    % Plotting the shear stress contour 
    Tal_yx = repmat(tau_yx, 1, n)';
    figure; 
    contourf(Y, X, Tal_yx, 20); % 20 contour levels
    colorbar;
    xlabel('Length of Plates (x)');
    ylabel('Height between Plates (y)');
    title('Stress Contour Plot');
    grid on;
end
