classdef Copy_of_MATSCF_noDImTab_exported < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure                        matlab.ui.Figure
        TabGroup                        matlab.ui.container.TabGroup
        HomeTab                         matlab.ui.container.Tab
        SupervisedbyDavidBuchnerandDrRonnyPiniLabel_2  matlab.ui.control.Label
        Hyperlink2                      matlab.ui.control.Hyperlink
        Contactangelinajia22imperialacukLabel  matlab.ui.control.Label
        Hyperlink                       matlab.ui.control.Hyperlink
        Label_4                         matlab.ui.control.Label
        Image                           matlab.ui.control.Image
        SupervisedbyDavidBuchnerandDrRonnyPiniLabel  matlab.ui.control.Label
        ByAngelinaJiaLabel              matlab.ui.control.Label
        subdescriptionherekeptshortcanbefurtherexplainedinalinkLabel  matlab.ui.control.Label
        ModellingandAnalysisToolforSteadyChannelFlowMATSCFLabel  matlab.ui.control.Label
        SIMPLETab                       matlab.ui.container.Tab
        ReEditField                     matlab.ui.control.NumericEditField
        VelocityQuiverPlotSwitch        matlab.ui.control.Switch
        VelocityQuiverPlotSwitchLabel   matlab.ui.control.Label
        PressureContourSwitch           matlab.ui.control.Switch
        PressureContourSwitchLabel      matlab.ui.control.Label
        ReEditFieldLabel                matlab.ui.control.Label
        exportWorkspace                 matlab.ui.control.Button
        Gsimple                         matlab.ui.control.NumericEditField
        pressuregradientforUinletLabel  matlab.ui.control.Label
        maxVelSIMPLE                    matlab.ui.control.NumericEditField
        maxvelocityinthexdirectionEditFieldLabel  matlab.ui.control.Label
        updated                         matlab.ui.control.Button
        DynamicViscositykgm1s1EditField  matlab.ui.control.NumericEditField
        DynamicViscositykgm1s1EditFieldLabel  matlab.ui.control.Label
        HeightmEditField                matlab.ui.control.NumericEditField
        HeightmEditFieldLabel_2         matlab.ui.control.Label
        LengthmEditField                matlab.ui.control.NumericEditField
        LengthmEditFieldLabel           matlab.ui.control.Label
        Densitykgm3EditField            matlab.ui.control.NumericEditField
        Densitykgm3EditFieldLabel       matlab.ui.control.Label
        Profile_4                       matlab.ui.control.NumericEditField
        ProfileatxEditField_4Label      matlab.ui.control.Label
        Profile_3                       matlab.ui.control.NumericEditField
        ProfileatxEditField_3Label      matlab.ui.control.Label
        Profile_2                       matlab.ui.control.NumericEditField
        ProfileatxEditField_2Label      matlab.ui.control.Label
        Profile_1                       matlab.ui.control.NumericEditField
        ProfileatxEditFieldLabel        matlab.ui.control.Label
        LowerPlateVelocitymsEditField_2  matlab.ui.control.NumericEditField
        LowerPlateVelocitymsEditField_2Label  matlab.ui.control.Label
        UpperPlateVelocitymsEditField_2  matlab.ui.control.NumericEditField
        UpperPlateVelocitymsEditField_2Label  matlab.ui.control.Label
        InletVelocitymsEditField        matlab.ui.control.NumericEditField
        InletVelocitymsEditFieldLabel   matlab.ui.control.Label
        GenerateButton                  matlab.ui.control.Button
        SIMPLEAlgorithmNavierStokesSolverforSteadyFlowLabel  matlab.ui.control.Label
        UIAxes_overall                  matlab.ui.control.UIAxes
        UIAxes_v                        matlab.ui.control.UIAxes
        UIAxes_4                        matlab.ui.control.UIAxes
        UIAxes_3                        matlab.ui.control.UIAxes
        UIAxes_2                        matlab.ui.control.UIAxes
        UIAxes_1                        matlab.ui.control.UIAxes
        UIAxes_u                        matlab.ui.control.UIAxes
        AnalyticalTab                   matlab.ui.container.Tab
        ReEditField_Ana                 matlab.ui.control.NumericEditField
        ReEditField_2Label              matlab.ui.control.Label
        AverageVelocityEditField        matlab.ui.control.NumericEditField
        AverageVelocityEditFieldLabel   matlab.ui.control.Label
        exportAnalytical                matlab.ui.control.Button
        updated_2                       matlab.ui.control.Button
        KeyResultsLabel                 matlab.ui.control.Label
        Label_6                         matlab.ui.control.Label
        YPositionofminmEditField        matlab.ui.control.NumericEditField
        YPositionofminmEditFieldLabel   matlab.ui.control.Label
        MinVelocitymsEditField          matlab.ui.control.NumericEditField
        MinVelocitymsEditFieldLabel     matlab.ui.control.Label
        YPositionofmaxmEditField        matlab.ui.control.NumericEditField
        MaxVelocitymsEditField_2Label_2  matlab.ui.control.Label
        MaxVelocitymsEditField          matlab.ui.control.NumericEditField
        MaxVelocitymsLabel              matlab.ui.control.Label
        GenerateButton_2                matlab.ui.control.Button
        XdirPressureGradientPamEditField  matlab.ui.control.NumericEditField
        XdirPressureGradientPamEditFieldLabel  matlab.ui.control.Label
        LowerPlateVelocitymsEditField   matlab.ui.control.NumericEditField
        LowerPlateVelocitymsEditFieldLabel  matlab.ui.control.Label
        UpperPlateVelocitymsEditField   matlab.ui.control.NumericEditField
        UpperPlateVelocitymsEditFieldLabel  matlab.ui.control.Label
        DynamicViscosityEditField       matlab.ui.control.NumericEditField
        DynamicViscosityPasEditFieldLabel  matlab.ui.control.Label
        HeightEditField                 matlab.ui.control.NumericEditField
        HeightmEditFieldLabel           matlab.ui.control.Label
        Label                           matlab.ui.control.Label
        UIAxes4                         matlab.ui.control.UIAxes
        UIAxes2                         matlab.ui.control.UIAxes
        UIAxes3                         matlab.ui.control.UIAxes
        UIAxes1                         matlab.ui.control.UIAxes
        SettingsTab                     matlab.ui.container.Tab
        RESETALLButton                  matlab.ui.control.Button
        gridPtsAnalytical               matlab.ui.control.NumericEditField
        gridpointsinydirectionEditField_2Label  matlab.ui.control.Label
        AnalyticalTabLabel              matlab.ui.control.Label
        SIMPLETabLabel                  matlab.ui.control.Label
        StreamlinesOnOff                matlab.ui.control.Switch
        StreamlinesoncontourplotsSwitchLabel  matlab.ui.control.Label
        gridpointsinydirectionEditField  matlab.ui.control.NumericEditField
        gridpointsinydirectionEditFieldLabel  matlab.ui.control.Label
    end

    properties (Access = public)
        % Description
        %making properties so can access variable values in other
        %functions, especially if want to have isolated graphs
        N_points_y = 30;
        N_points_x;
        U_final;
        V_final;
        P_final;
        Y_lin;
        X_lin;
        Min_vel;
        Max_vel;
        Dom_length_y;
        pressure_on= 'Off'; %***
        quiver_on= 'Off';
        streamlines_on= 'On';
        Reynolds;
        SIM_mu;
        SIM_rho;
        SIM_Uin;
        SIM_H;
        grid_spacing;
        x_domain;
        y_domain;
        fullyDevSIM;
        G_term_SIM;
        length; 
        

        %for analytical tab 
        u_Analytical;
        tal_Analytical;
        H_Analytical;
        y_Analytical;
        max_u_Analytical;
        y_max_Analytical;
        min_u_Analytical;
        y_min_Analytical;
        Re_Analytical;
        grid_pts_Analytical=150;

        %for dimensionless tab

        Dist;
        is_updated = 0; %0 is updated, 1 is not updated
    end

    methods (Access = private)

        function results =SIMPLEfunc(app, length, breadth, top_vel, bot_vel, inlet_vel, x_target1, x_target2, x_target3, x_target4, muFact, rhoFact, y_points, streamlines_on);

            %% Defining the problem domain

            % Physical domain dimensions
            %dimensionless values, length is physical length/phys height
            %so height will be 1, and length scaled to it
            % Note: The critial Re number for Plane Poiseuille Flow is 1000-2000
            % make sure this is not exceeded, as this is a laminar solver and does
            % otherwise diverge
            L_dom = length; % [m] domain length
            H_dom = breadth; % [m] domain height
            rho_phy = rhoFact; % [kg/m3] physical density of fluid
            mu_phy = muFact; % [kg m-1 s-1] pysical dynamic viscosity of fluid
            U_in = inlet_vel;  % [m/s] inlet velocity

            % Reference values for non-dimensionalization
            rho_ref = 1000; % [kg/m3] density of water
            mu_ref = 0.001; % [kg m-1 s-1] dynamic viscosity of water

            Re_ref = rho_ref*U_in*H_dom/mu_ref; % [-] reference Reynolds number

            dom_length_x = L_dom/H_dom; %dimensionless length/heights?
            dom_length_y = H_dom/H_dom;
            % Spatial discretization, Number of grid cells
            n_points_y = y_points;
            n_points_x = (dom_length_x/dom_length_y)*(n_points_y);
            app.N_points_x = n_points_x;
            
            %{
            dom_length_x = length;
            dom_length_y = breadth;
             % Spatial discretization, Number of grid cells
            n_points_y = y_points;
            n_points_x =  dom_length_x*(n_points_y);
            app.N_points_x= n_points_x;
            %}

            top_vel = top_vel;
            bot_vel = bot_vel;
            inlet_u=inlet_vel;

            is_negative=0;
            if inlet_u <0
                is_negative = 1;
                inlet_u = inlet_u*(-1); %change sign of inlet so converges
                top_vel = top_vel*(-1); %flip top velocity
                bot_vel = bot_vel*(-1); %flip bottom velocity
            end


            h = dom_length_y/(n_points_y-1); %grid spacing
            app.grid_spacing=h;
           %{
            % Fluid properties, change with reynolds num
            mu = muFact; %dynamic visco
            rho = rhoFact; %non dim density
           %}
            % Fluid properties, change with reynolds num
            mu = 1/Re_ref; % non-dimensional factor/viscosity
            rho = rho_phy/rho_ref; %non dimensionalized density
            % Under-relaxation factors
            alpha = 0.1;
            alpha_p = 0.8;

            %% Initializing the variables
            % Final collocated variables / Physical variables
            u_final(n_points_y,n_points_x) = 0;
            v_final(n_points_y,n_points_x) = 0;
            p_final(n_points_y,n_points_x) = 0;
            u_final(:,1) = inlet_u;
            u_final(:,n_points_x) = inlet_u;

            % Staggered variables / Numerical variables
            u(n_points_y+1,n_points_x) = 0;
            ustar(n_points_y+1,n_points_x) = 0;
            d_u(n_points_y+1,n_points_x) = 0;
            v(n_points_y,n_points_x+1) = 0;
            vstar(n_points_y,n_points_x+1) = 0;
            d_v(n_points_y,n_points_x+1) = 0;
            p(n_points_y+1,n_points_x+1) = 0;
            pstar(n_points_y+1,n_points_x+1) = 0;
            pcor(n_points_y+1,n_points_x+1) = 0;
            b(n_points_y+1,n_points_x+1) = 0;
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
            figure('Name', 'Residuals');

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
                        ustar(i,j) = (1-alpha)*u(i,j) + alpha*ustar_mid;
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
                        vstar(i,j) = (1-alpha)*v(i,j) + alpha*vstar_mid;
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

                % Correcting the velocities
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

                % continuity eq. as error measure
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

            % Mapping the staggered variables to collocated variables
            for i = 1:n_points_y
                for j = 1:n_points_x
                    u_final(i,j) = 0.5*(u(i,j) + u(i+1,j));
                    v_final(i,j) = 0.5*(v(i,j) + v(i,j+1));
                    p_final(i,j) = 0.25*(p(i,j) + p(i,j+1) + p(i+1,j) + p(i+1,j+1));
                end
            end

            %% Contour and vector visuals.
            x_dom = (((1:n_points_x)-1).*h).*H_dom;
            y_dom = (1-((1:n_points_y)-1).*h).*H_dom;
            app.x_domain = x_dom;
            app.y_domain = y_dom;
            [X,Y] = meshgrid(x_dom,y_dom);
            %***
            if is_negative ==1
                u_average = -sum(u_final(:,1)) / n_points_y;
            else  
                u_average = sum(u_final(:,n_points_x)) / n_points_y;
            end
            G_term  = ((((top_vel-bot_vel)/2)+bot_vel-u_average)*12*mu_phy)/(H_dom^2);
            %G_term_rounded = round(G_term, 5 - floor(log10(abs(G_term))) - 1);
            %app.G_term_SIM = G_term_rounded;
            app.G_term_SIM = G_term;
            %** is_negative =1, need to invert the x-axis
            if is_negative ==1
                u_final = -1*flip(u_final,2); %columns should be flipped, but rows remain
                v_final = flip(v_final,2);
                p_final = flip(p_final,2);
            end

            %! updating the properties so profiles can be used throughout
            %app
            app.U_final = u_final;
            app.V_final = v_final;
            app.P_final = p_final;
               
            %u-velocity contour
            contourf(app.UIAxes_u, X,Y, u_final, 21, 'LineStyle', 'none')
            colorbar(app.UIAxes_u)
            colormap(app.UIAxes_u,'jet')
            xlabel(app.UIAxes_u,'x')
            ylabel(app.UIAxes_u,'y')
            hold on
           if strcmp(streamlines_on, 'On')
                streamslice(app.UIAxes_u,X, Y, u_final, v_final, 0.5,'noarrows','k');
            end
            pbaspect(app.UIAxes_u,[1.2 1 2])
            hold off;

            %v-velocity contour
            contourf(app.UIAxes_v, X,Y, v_final, 21, 'LineStyle', 'none')
            colorbar(app.UIAxes_v)
            colormap(app.UIAxes_v,'jet')
            xlabel(app.UIAxes_v,'x')
            ylabel(app.UIAxes_v,'y')
            hold on
            if strcmp(streamlines_on, 'On')
            streamslice(app.UIAxes_v, X, Y, u_final, v_final, 0.5,'noarrows','k');
            end
            pbaspect(app.UIAxes_v,[1.2 1 2])
            hold off;

            %overall velocity contour
            vel_mag = sqrt(u_final.^2+v_final.^2);
            contourf(app.UIAxes_overall,X,Y, vel_mag, 21, 'LineStyle', 'none')
            colorbar(app.UIAxes_overall)
            colormap(app.UIAxes_overall,'jet')
            xlabel(app.UIAxes_overall,'x')
            ylabel(app.UIAxes_overall,'y')
            hold on
           if strcmp(streamlines_on, 'On')
            streamslice(app.UIAxes_overall,X, Y, u_final, v_final, 0.5,'noarrows','k');
            end
            pbaspect(app.UIAxes_overall,[1 1 2])
            hold off;
            
            %find a way to make these optional graphs to see
            %pressure contour
            if strcmp(app.pressure_on, 'On') 
                figure('Name', 'Pressure Contour Plot')
                contourf(X,Y, p_final, 21, 'LineStyle', 'none')
                colorbar
                colormap('jet')
                xlabel('x')
                ylabel('y')
                 pbaspect([1.2 1 2])
            end

            %quiver plot for velocities
            if strcmp(app.quiver_on,'On')
                figure('Name', 'Velocity Quiver Plot');
                hold on
                quiver(X, Y, u_final, v_final, 1, 'k')
                pbaspect([1.2 1 2])
            end

            %% Plotting the profile evolution
            %used the u-vel, coudl also use overall mag, but doesn't show uniform at 0

            % set up
            y_lin = linspace(dom_length_y,0,n_points_y).*H_dom;
            x_lin= linspace(0,dom_length_x,n_points_x).*H_dom;
            max_vel = max(u_final(:,n_points_x));
            min_vel = min(u_final(:,n_points_x));
            app.fullyDevSIM = u_final(:,n_points_x);
           %{
            % Find the index corresponding to the maximum velocity
            [max_vel, max_index] = max(u_final(:,n_points_x));

            % Calculate the corresponding y-coordinate
            y_max_position = (max_index - 1) * hy;
            disp(y_max_position);
           %}
            %** adjust values if inlet is on oppostie side
            if is_negative ==1
                max_vel = max(u_final(:,1));
                min_vel = min(u_final(:,1));
            end
            %access to propoerties outside of the function
            app.Max_vel= max_vel;
            app.Min_vel= min_vel;
            app.Dom_length_y= dom_length_y;
            app.Y_lin = y_lin;
            app.X_lin = x_lin;

            %graphs, x_target can change, esp in UI later on
            x_target =x_target1;
            [~, xIndex] = min(abs(x_lin - x_target));
            Vel = u_final(:, xIndex);
            plot(app.UIAxes_1, Vel, y_lin, '-');
            xlim(app.UIAxes_1,[min_vel-.001,max_vel]);
            ylim(app.UIAxes_1,[0,H_dom]);
            grid (app.UIAxes_1, 'on');

            x_target = x_target2;
            [~, xIndex] = min(abs(x_lin - x_target));
            Vel = u_final(:, xIndex);
            plot(app.UIAxes_2, Vel, y_lin, '-');
            xlim(app.UIAxes_2,[min_vel-0.001,max_vel]);
            ylim(app.UIAxes_2,[0,H_dom]);
            grid (app.UIAxes_2,'on');

            x_target = x_target3;
            [~, xIndex] = min(abs(x_lin - x_target));
            Vel = u_final(:, xIndex);
            plot(app.UIAxes_3,Vel, y_lin, '-');
            xlim(app.UIAxes_3,[min_vel-.001,max_vel]);
            ylim(app.UIAxes_3,[0,H_dom]);
            grid (app.UIAxes_3,'on');

            x_target = x_target4;
            [~, xIndex] = min(abs(x_lin - x_target));
            Vel = u_final(:, xIndex);
            plot(app.UIAxes_4, Vel, y_lin, '-');
            xlim(app.UIAxes_4,[min_vel-.001,max_vel]);
            ylim(app.UIAxes_4,[0,H_dom]);
            grid (app.UIAxes_4,'on');
        end


        function results = FullyDevFunc(app, mu, H, G, u1, u2)
            n = app.grid_pts_Analytical; %number of grid points
            y = linspace(0, H, n)'; % y discretized into n points
            dist = y;
            app.Dist = dist;

            % Analytical solution for velocity profile
            u = (G/(2*mu)) * (y.^2 - H*y) + ((u2 - u1)/H) * y + u1;

            % Analytical solution for shear stress profile
            % Shear stress tau_yx = mu * du/dy
            % Analytical derivative of u(y):
            du_dy = (G/(2*mu)) * (2*y - H) + (u2 - u1)/H;
            tau_yx = mu * du_dy;

            % Find max/min value and its index
            [max_u, index] = max(u);
            [min_u, minIndex] = min(u);

            % y-coordinate corresponding to the max/min value
            y_max = dist(index);
            y_min = dist(minIndex);

            % Format the numbers to 4 significant figures
            max_u_4sf = sprintf('%.4g', max_u);
            y_max_4sf = sprintf('%.2g', y_max);
            min_u_4sf = sprintf('%.4g', min_u);
            y_min_4sf = sprintf('%.2g', y_min);
            

            %assigning property values for analytical tab
            app.grid_pts_Analytical = n;
            app.u_Analytical = u;
            app.tal_Analytical = tau_yx;
            app.H_Analytical = H;
            app.y_Analytical = y;
            app.max_u_Analytical = max_u_4sf;
            app.y_max_Analytical = y_max_4sf;
            app.min_u_Analytical = min_u_4sf;
            app.y_min_Analytical = y_min_4sf;
        end
    end


    % Callbacks that handle component events
    methods (Access = private)

        % Code that executes after component creation
        function startupFcn(app)
         
            app.SIM_rho = app.Densitykgm3EditField.Value;
            app.SIM_H = app.HeightmEditField.Value;
            app.SIM_Uin= app.InletVelocitymsEditField.Value;
            app.SIM_mu = app.DynamicViscositykgm1s1EditField.Value;
            app.Reynolds = abs(app.SIM_rho*app.SIM_H*app.SIM_Uin/app.SIM_mu);
            app.ReEditField.Value = app.Reynolds;
            app.grid_pts_Analytical = app.gridPtsAnalytical.Value;
        end

        % Button pushed function: GenerateButton
        function GenerateButtonPushed2(app, event)
            app.GenerateButton.BackgroundColor = [1.00, 0.41, 0.16];
            disp('Button 1 pressed'); % Debugging statement
            disp('Accessing properties...'); % Debugging statement
            app.pressure_on = app.PressureContourSwitch.Value;
            app.quiver_on = app.VelocityQuiverPlotSwitch.Value;
            streamlines_on = app.StreamlinesOnOff.Value;
            app.length = app.LengthmEditField.Value;
            app.SIM_H = app.HeightmEditField.Value;
            top_vel = app.UpperPlateVelocitymsEditField_2.Value;
            bot_vel = app.LowerPlateVelocitymsEditField_2.Value;
            inlet_vel = app.InletVelocitymsEditField.Value;
            x_target1 = app.Profile_1.Value;
            x_target2 = app.Profile_2.Value;
            x_target3 = app.Profile_3.Value;
            x_target4 = app.Profile_4.Value;
            muFact= app.DynamicViscositykgm1s1EditField.Value;
            rhoFact= app.Densitykgm3EditField.Value;
            y_points = app.gridpointsinydirectionEditField.Value;
            %input here the run code function
            SIMPLEfunc(app, app.length, app.SIM_H, top_vel, bot_vel, inlet_vel, x_target1, x_target2, x_target3, x_target4, muFact, rhoFact, y_points, streamlines_on);
            app.maxVelSIMPLE.Value = app.Max_vel; 
            if inlet_vel <0
                 app.maxVelSIMPLE.Value = app.Min_vel;
            end
            app.Gsimple.Value = app.G_term_SIM;
            disp('done');
            app.GenerateButton.BackgroundColor = [0.39, 0.83, 0.07];
            app.updated.BackgroundColor = [0.39, 0.83, 0.07];
        end

        % Value changed function: Profile_2
        function Profile_2ValueChanged2(app, event)
            value2 = app.Profile_2.Value;
             if value2 <0
                value2 = 0;
                app.Profile_2.Value = value2;
            elseif value2 > app.length
                value2 = app.length;
                app.Profile_2.Value = value2;
            end
            x_target = value2;
            [~, xIndex] = min(abs(app.X_lin - x_target));
            Vel = app.U_final(:, xIndex);
            plot(app.UIAxes_2, Vel, app.Y_lin, '-');
            xlim(app.UIAxes_2,[app.Min_vel-.001,app.Max_vel]);
            ylim(app.UIAxes_2,[0,app.HeightmEditField.Value]);
            grid (app.UIAxes_2, 'on');
        end

        % Value changed function: Profile_3
        function Profile_3ValueChanged2(app, event)
            value3 = app.Profile_3.Value;
             if value3 <0
                value3 = 0;
                app.Profile_3.Value = value3;
            elseif value3 > app.length
                value3 = app.length;
                app.Profile_3.Value = value3;
            end
            x_target = value3;
            [~, xIndex] = min(abs(app.X_lin - x_target));
            Vel = app.U_final(:, xIndex);
            plot(app.UIAxes_3, Vel, app.Y_lin, '-');
            xlim(app.UIAxes_3,[app.Min_vel-.001,app.Max_vel]);
            ylim(app.UIAxes_3,[0,app.HeightmEditField.Value]);
            grid (app.UIAxes_3, 'on');
        end

        % Value changed function: Profile_4
        function Profile_4ValueChanged2(app, event)
            value4 = app.Profile_4.Value;
             if value4 <0
                value4 = 0;
                app.Profile_4.Value = value4;
            elseif value4 > app.length
                value4 = app.length;
                app.Profile_4.Value = value4;
            end
            x_target = value4;
            [~, xIndex] = min(abs(app.X_lin - x_target));
            Vel = app.U_final(:, xIndex);
            plot(app.UIAxes_4, Vel, app.Y_lin, '-');
            xlim(app.UIAxes_4,[app.Min_vel-.001,app.Max_vel]);
            ylim(app.UIAxes_4,[0,app.HeightmEditField.Value]);
            grid (app.UIAxes_4, 'on');
        end

        % Value changed function: Profile_1
        function Profile_1ValueChanged2(app, event)
            value1 = app.Profile_1.Value;
            if value1 <0
                value1 = 0;
                app.Profile_1.Value = value1;
            elseif value1 > app.length
                value1 = app.length;
                app.Profile_1.Value = value1;
            end
            x_target = value1;
            [~, xIndex] = min(abs(app.X_lin - x_target));
            Vel = app.U_final(:, xIndex);
            plot(app.UIAxes_1, Vel, app.Y_lin, '-');
            xlim(app.UIAxes_1,[app.Min_vel-.001,app.Max_vel]);
            ylim(app.UIAxes_1,[0,app.HeightmEditField.Value]);
            grid (app.UIAxes_1, 'on');
        end

        % Button pushed function: GenerateButton_2
        function GenerateButton_2Pushed(app, event)
            app.GenerateButton_2.BackgroundColor = [1.00, 0.41, 0.16];
            disp('Button 2 pressed'); % Debugging statement
            mu = app.DynamicViscosityEditField.Value;
            H = app.HeightEditField.Value;
            G = app.XdirPressureGradientPamEditField.Value;
            u2 = app.UpperPlateVelocitymsEditField.Value;
            u1 = app.LowerPlateVelocitymsEditField.Value;
            %call the function
            FullyDevFunc(app, mu, H, G, u1, u2);
            disp('FullyDevFunc done');

            % Plotting the velocity profile
            plot(app.UIAxes1, app.u_Analytical, app.Dist, '-');
            xlabel(app.UIAxes1, 'Velocity of Flowing Fluid (m/s)');
            ylabel(app.UIAxes1,'Distance between the plates(m)');
            grid (app.UIAxes1,'on');
            hold (app.UIAxes1,'on');
            yline(app.UIAxes1,H/2, '--');
            xline(app.UIAxes1,0, '--');
            hold (app.UIAxes1,'off');

            % Plotting the shear stress profile
            plot(app.UIAxes3, app.tal_Analytical, app.Dist, '-');
            xlabel(app.UIAxes3,'Shear Stress of Flowing Fluid(Pa)');
            ylabel(app.UIAxes3,'Distance between the plates(m)');
            grid (app.UIAxes3,'on');
            hold (app.UIAxes3,'on');
            yline(app.UIAxes3,app.H_Analytical/2, '--');
            xline(app.UIAxes3,0,'--');
            hold (app.UIAxes3,'off');
            disp('Done');

            % Define the length of the plates in the x-direction, since 2D for
            % contour plot
            L = 1; % assuming a length of 1 meter
            x = linspace(0, L, app.grid_pts_Analytical); % x-coordinates
            [X, Y] = meshgrid(x, app.y_Analytical); % creating a mesh grid with x and y dimensions

            % Replicate the velocity profile across the x-direction
            U = repmat(app.u_Analytical, 1,  app.grid_pts_Analytical)'; % Replicates u as rows of 2D matrix, ensure proper orientation with '

            % Plotting the velocity contour
            contourf(app.UIAxes2,Y, X, U, 20, 'LineColor', 'none'); % 20 contour levels
            colorbar(app.UIAxes2);
            xlabel(app.UIAxes2,'Length of Plates (x)');
            ylabel(app.UIAxes2,'Height between Plates (y)');

            % Plotting the shear stress contour
            Tal_yx = repmat(app.tal_Analytical, 1,  app.grid_pts_Analytical)';
            contourf(app.UIAxes4, Y, X, Tal_yx, 20,'LineColor', 'none'); % 20 contour levels
            colorbar(app.UIAxes4);
            xlabel(app.UIAxes4,'Length of Plates (x)');
            ylabel(app.UIAxes4,'Height between Plates (y)');

            app.MaxVelocitymsEditField.Value = str2double(app.max_u_Analytical);
            app.YPositionofmaxmEditField.Value = str2double(app.y_max_Analytical);
            app.MinVelocitymsEditField.Value = str2double(app.min_u_Analytical);
            app.YPositionofminmEditField.Value = str2double(app.y_min_Analytical);
            Analytical_average = sum(app.u_Analytical) / app.grid_pts_Analytical;
            %Analytical_average_rounded = round(Analytical_average, 3 - floor(log10(abs(Analytical_average))) - 1);
            %app.AverageVelocityEditField.Value = Analytical_average_rounded;
            app.AverageVelocityEditField.Value = Analytical_average;
            app.GenerateButton_2.BackgroundColor = [0.39, 0.83, 0.07];
            app.updated_2.BackgroundColor = [0.39, 0.83, 0.07];
            ReAnalytical = 1000* Analytical_average*H/mu;
            app.ReEditField_Ana.Value = ReAnalytical;
            if ReAnalytical >= 3000
                app.ReEditField_Ana.FontColor = [1.00,0.00,0.00];
            elseif ReAnalytical < 3000
                app.ReEditField_Ana.FontColor = [0.11,0.68,0.10];
            end 

        end

        % Value changed function: InletVelocitymsEditField
        function InletVelocitymsEditFieldValueChanged(app, event)
            app.SIM_Uin = app.InletVelocitymsEditField.Value;
            app.Reynolds = abs(app.SIM_rho*app.SIM_H*app.SIM_Uin/app.SIM_mu);
            app.ReEditField.Value = app.Reynolds;
            if app.Reynolds >= 3000
                app.ReEditField.FontColor = [1.00,0.00,0.00];
            elseif app.Reynolds < 3000
                app.ReEditField.FontColor = [0.11,0.68,0.10];
            end 
            app.updated.BackgroundColor = [1.00,0.00,0.00];
        end

        % Value changed function: UpperPlateVelocitymsEditField_2
        function UpperPlateVelocitymsEditField_2ValueChanged(app, event)
            value = app.UpperPlateVelocitymsEditField_2.Value;
            app.Reynolds = abs(app.SIM_rho*app.SIM_H*app.SIM_Uin/app.SIM_mu);
            app.ReEditField.Value = app.Reynolds;
            if app.Reynolds >= 3000
                app.ReEditField.FontColor = [1.00,0.00,0.00];
            elseif app.Reynolds < 3000
                app.ReEditField.FontColor = [0.11,0.68,0.10];
            end 
            app.updated.BackgroundColor = [1.00,0.00,0.00];
        end

        % Value changed function: LowerPlateVelocitymsEditField_2
        function LowerPlateVelocitymsEditField_2ValueChanged(app, event)
            value = app.LowerPlateVelocitymsEditField_2.Value;
           app.Reynolds = abs(app.SIM_rho*app.SIM_H*app.SIM_Uin/app.SIM_mu);
            app.ReEditField.Value = app.Reynolds;
            if app.Reynolds >= 3000
                app.ReEditField.FontColor = [1.00,0.00,0.00];
            elseif app.Reynolds < 3000
                app.ReEditField.FontColor = [0.11,0.68,0.10];
            end 
             app.updated.BackgroundColor = [1.00,0.00,0.00];
        end

        % Value changed function: Densitykgm3EditField
        function Densitykgm3EditFieldValueChanged(app, event)
            app.SIM_rho = app.Densitykgm3EditField.Value;
            app.Reynolds = abs(app.SIM_rho*app.SIM_H*app.SIM_Uin/app.SIM_mu);
            app.ReEditField.Value = app.Reynolds;
            if app.Reynolds >= 3000
                app.ReEditField.FontColor = [1.00,0.00,0.00];
            elseif app.Reynolds < 3000
                app.ReEditField.FontColor = [0.11,0.68,0.10];
            end 
             app.updated.BackgroundColor = [1.00,0.00,0.00];
        end

        % Value changed function: DynamicViscositykgm1s1EditField
        function DynamicViscositykgm1s1EditFieldValueChanged(app, event)
            app.SIM_mu = app.DynamicViscositykgm1s1EditField.Value;
           app.Reynolds = abs(app.SIM_rho*app.SIM_H*app.SIM_Uin/app.SIM_mu);
            app.ReEditField.Value = app.Reynolds;
            if app.Reynolds >= 3000
                app.ReEditField.FontColor = [1.00,0.00,0.00];
            elseif app.Reynolds < 3000
                app.ReEditField.FontColor = [0.11,0.68,0.10];
            end 
             app.updated.BackgroundColor = [1.00,0.00,0.00];
        end

        % Value changed function: HeightmEditField
        function HeightmEditFieldValueChanged(app, event)
            app.SIM_H = app.HeightmEditField.Value;
            app.Reynolds = abs(app.SIM_rho*app.SIM_H*app.SIM_Uin/app.SIM_mu);
            app.ReEditField.Value = app.Reynolds;
            if app.Reynolds >= 3000
                app.ReEditField.FontColor = [1.00,0.00,0.00];
            elseif app.Reynolds < 3000
                app.ReEditField.FontColor = [0.11,0.68,0.10];
            end 
             app.updated.BackgroundColor = [1.00,0.00,0.00];
        end

        % Value changed function: LengthmEditField
        function LengthmEditFieldValueChanged(app, event)
            value = app.LengthmEditField.Value;
             app.updated.BackgroundColor = [1.00,0.00,0.00];
        end

        % Value changed function: HeightEditField
        function HeightEditFieldValueChanged(app, event)
            value = app.HeightEditField.Value;
             app.updated_2.BackgroundColor = [1.00,0.00,0.00];
        end

        % Value changed function: DynamicViscosityEditField
        function DynamicViscosityEditFieldValueChanged(app, event)
            value = app.DynamicViscosityEditField.Value;
            app.updated_2.BackgroundColor = [1.00,0.00,0.00];
        end

        % Value changed function: UpperPlateVelocitymsEditField
        function UpperPlateVelocitymsEditFieldValueChanged(app, event)
            value = app.UpperPlateVelocitymsEditField.Value;
            app.updated_2.BackgroundColor = [1.00,0.00,0.00];
        end

        % Value changed function: LowerPlateVelocitymsEditField
        function LowerPlateVelocitymsEditFieldValueChanged(app, event)
            value = app.LowerPlateVelocitymsEditField.Value;
            app.updated_2.BackgroundColor = [1.00,0.00,0.00];
        end

        % Value changed function: XdirPressureGradientPamEditField
        function XdirPressureGradientPamEditFieldValueChanged(app, event)
            value = app.XdirPressureGradientPamEditField.Value;
            app.updated_2.BackgroundColor = [1.00,0.00,0.00];
        end

        % Button pushed function: exportWorkspace
        function exportWorkspaceButtonPushed(app, event)
            disp('Variables exported to workspace'); 
            assignin('base', 'u_velocity_field', app.U_final);
            assignin('base', 'v_velocity_field', app.V_final);
            assignin('base', 'pressure_field', app.P_final);
            assignin('base', 'n_points_y', app.N_points_y);
            assignin('base', 'n_points_x', app.N_points_x);
            assignin('base', 'rho', app.SIM_rho);
            assignin('base', 'mu_dynamic', app.SIM_mu);
            assignin('base', 'x_dom', app.x_domain);
            assignin('base', 'y_dom', app.y_domain);
            assignin('base', 'grid_spacing', app.grid_spacing);   
        end

        % Button pushed function: exportAnalytical
        function exportAnalyticalButtonPushed(app, event)
            disp('Variables exported to workspace'); 
            assignin('base', 'u_analytical', app.u_Analytical);
            assignin('base', 'tal_analytical', app.tal_Analytical);
            assignin('base', 'y_analytical', app.y_Analytical);
            assignin('base', 'H_analytical', app.H_Analytical);
            assignin('base', 'max_u_analytical', app.max_u_Analytical);
            assignin('base', 'y_max_analytical', app.y_max_Analytical);
            assignin('base', 'min_u_analytical', app.min_u_Analytical);
            assignin('base', 'y_min_analytical', app.y_min_Analytical);
            assignin('base', 'grid_points_analytical', app.grid_pts_Analytical);
        end

        % Value changed function: ReEditField
        function ReEditFieldValueChanged(app, event)
            value = app.ReEditField.Value;
            if value >= 3000
                app.ReEditField.FontColor = [1.00,0.00,0.00];
            elseif value < 3000
                app.ReEditField.FontColor = [0.11,0.68,0.10];
            end 
        end

        % Value changed function: gridPtsAnalytical
        function gridPtsAnalyticalValueChanged(app, event)
            value = app.gridPtsAnalytical.Value;
            app.grid_pts_Analytical = value;
        end

        % Button pushed function: RESETALLButton
        function RESETALLButtonPushed(app, event)
            choice = uiconfirm(app.UIFigure, ...
                'Are you sure you want to reset the app?', ...
                'Confirm Reset', ...
                'Options', {'Yes', 'No'}, ...
                'DefaultOption', 2, ...
                'CancelOption', 2);

            % Take action based on the user's choice
            if strcmp(choice, 'Yes')
                disp('resetting');
                %clear all of the graphs
                cla(app.UIAxes_u); %simple tab
                cla(app.UIAxes_1);
                cla(app.UIAxes_2);
                cla(app.UIAxes_3);
                cla(app.UIAxes_4);
                cla(app.UIAxes_v);
                cla(app.UIAxes_overall);
                cla(app.UIAxes1);
                cla(app.UIAxes3);
                cla(app.UIAxes2);
                cla(app.UIAxes4);

                %reset tab field values
                app.InletVelocitymsEditField.Value = 0.01;
                app.UpperPlateVelocitymsEditField_2.Value = 0;
                app.LowerPlateVelocitymsEditField_2.Value = 0;
                app.ReEditField.Value = 1000;
                app.DynamicViscositykgm1s1EditField.Value = 0.001;
                app.HeightmEditField.Value = 0.1;
                app.LengthmEditField.Value =1;
                app.Densitykgm3EditField.Value = 1000;
                app.maxVelSIMPLE.Value = 0;
                app.Gsimple.Value = 0;
                app.ReEditField.FontColor = [0.11,0.68,0.10];

                app.AverageVelocityEditField.Value = 0;
                app.MinVelocitymsEditField.Value = 0;
                app.YPositionofminmEditField.Value = 0;
                app.MaxVelocitymsEditField.Value = 0;
                app.YPositionofmaxmEditField.Value = 0;
                app.XdirPressureGradientPamEditField.Value = -0.012;
                app.LowerPlateVelocitymsEditField.Value = 0;
                app.UpperPlateVelocitymsEditField.Value = 0;
                app.DynamicViscosityEditField.Value = 0.001;
                app.HeightEditField.Value = 0.1;

                %settings reset
                app.gridPtsAnalytical.Value = 150;
                app.StreamlinesOnOff.Value = 'On';
                app.gridpointsinydirectionEditField.Value = 30;
            end
                % Reset components to initial values
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Get the file path for locating images
            pathToMLAPP = fileparts(mfilename('fullpath'));

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Color = [1 1 1];
            app.UIFigure.Position = [100 100 836 573];
            app.UIFigure.Name = 'MATLAB App';

            % Create TabGroup
            app.TabGroup = uitabgroup(app.UIFigure);
            app.TabGroup.Position = [2 -12 873 586];

            % Create HomeTab
            app.HomeTab = uitab(app.TabGroup);
            app.HomeTab.Title = 'Home';
            app.HomeTab.BackgroundColor = [1 1 1];

            % Create ModellingandAnalysisToolforSteadyChannelFlowMATSCFLabel
            app.ModellingandAnalysisToolforSteadyChannelFlowMATSCFLabel = uilabel(app.HomeTab);
            app.ModellingandAnalysisToolforSteadyChannelFlowMATSCFLabel.HorizontalAlignment = 'center';
            app.ModellingandAnalysisToolforSteadyChannelFlowMATSCFLabel.FontSize = 28;
            app.ModellingandAnalysisToolforSteadyChannelFlowMATSCFLabel.FontWeight = 'bold';
            app.ModellingandAnalysisToolforSteadyChannelFlowMATSCFLabel.FontColor = [0 0 1];
            app.ModellingandAnalysisToolforSteadyChannelFlowMATSCFLabel.Position = [2 488 870 39];
            app.ModellingandAnalysisToolforSteadyChannelFlowMATSCFLabel.Text = 'Modelling and Analysis Tool for Steady Channel Flow(MATSCF)';

            % Create subdescriptionherekeptshortcanbefurtherexplainedinalinkLabel
            app.subdescriptionherekeptshortcanbefurtherexplainedinalinkLabel = uilabel(app.HomeTab);
            app.subdescriptionherekeptshortcanbefurtherexplainedinalinkLabel.HorizontalAlignment = 'center';
            app.subdescriptionherekeptshortcanbefurtherexplainedinalinkLabel.FontSize = 18;
            app.subdescriptionherekeptshortcanbefurtherexplainedinalinkLabel.Position = [1 464 871 23];
            app.subdescriptionherekeptshortcanbefurtherexplainedinalinkLabel.Text = 'An in-depth exploration of fluid flowing steadily between paralllel plates';

            % Create ByAngelinaJiaLabel
            app.ByAngelinaJiaLabel = uilabel(app.HomeTab);
            app.ByAngelinaJiaLabel.HorizontalAlignment = 'center';
            app.ByAngelinaJiaLabel.Position = [-1 438 873 22];
            app.ByAngelinaJiaLabel.Text = 'By Angelina Jia';

            % Create SupervisedbyDavidBuchnerandDrRonnyPiniLabel
            app.SupervisedbyDavidBuchnerandDrRonnyPiniLabel = uilabel(app.HomeTab);
            app.SupervisedbyDavidBuchnerandDrRonnyPiniLabel.HorizontalAlignment = 'center';
            app.SupervisedbyDavidBuchnerandDrRonnyPiniLabel.Position = [1 414 874 22];
            app.SupervisedbyDavidBuchnerandDrRonnyPiniLabel.Text = 'Supervised by David Buchner and Dr Ronny Pini';

            % Create Image
            app.Image = uiimage(app.HomeTab);
            app.Image.Position = [196 202 456 203];
            app.Image.ImageSource = fullfile(pathToMLAPP, 'ParallelPlateSystem (2).png');

            % Create Label_4
            app.Label_4 = uilabel(app.HomeTab);
            app.Label_4.HorizontalAlignment = 'center';
            app.Label_4.Position = [2 124 870 22];
            app.Label_4.Text = 'For user guidelines and further reading, use the links provided to view addional documentation';

            % Create Hyperlink
            app.Hyperlink = uihyperlink(app.HomeTab);
            app.Hyperlink.HorizontalAlignment = 'center';
            app.Hyperlink.Position = [-1 99 873 22];
            app.Hyperlink.Text = 'User manual ';

            % Create Contactangelinajia22imperialacukLabel
            app.Contactangelinajia22imperialacukLabel = uilabel(app.HomeTab);
            app.Contactangelinajia22imperialacukLabel.HorizontalAlignment = 'center';
            app.Contactangelinajia22imperialacukLabel.Position = [-1 24 871 22];
            app.Contactangelinajia22imperialacukLabel.Text = 'Contact: angelina.jia22@imperial.ac.uk';

            % Create Hyperlink2
            app.Hyperlink2 = uihyperlink(app.HomeTab);
            app.Hyperlink2.HorizontalAlignment = 'center';
            app.Hyperlink2.URL = 'https://github.com/angelinajia/Modeling-and-Analysis-of-Channel-Flow-.git';
            app.Hyperlink2.Position = [0 77 870 22];
            app.Hyperlink2.Text = 'Source Code(Git Repository)';

            % Create SupervisedbyDavidBuchnerandDrRonnyPiniLabel_2
            app.SupervisedbyDavidBuchnerandDrRonnyPiniLabel_2 = uilabel(app.HomeTab);
            app.SupervisedbyDavidBuchnerandDrRonnyPiniLabel_2.HorizontalAlignment = 'center';
            app.SupervisedbyDavidBuchnerandDrRonnyPiniLabel_2.WordWrap = 'on';
            app.SupervisedbyDavidBuchnerandDrRonnyPiniLabel_2.Position = [161 157 549 40];
            app.SupervisedbyDavidBuchnerandDrRonnyPiniLabel_2.Text = 'System definition used throughout the app, with u representing velocity in the x-direction, and v representing velocity in the y-direction';

            % Create SIMPLETab
            app.SIMPLETab = uitab(app.TabGroup);
            app.SIMPLETab.Title = 'SIMPLE';
            app.SIMPLETab.BackgroundColor = [1 1 1];

            % Create UIAxes_u
            app.UIAxes_u = uiaxes(app.SIMPLETab);
            title(app.UIAxes_u, 'X-velocity Contours ')
            xlabel(app.UIAxes_u, 'X')
            ylabel(app.UIAxes_u, 'Y')
            zlabel(app.UIAxes_u, 'Z')
            app.UIAxes_u.Position = [161 224 228 184];

            % Create UIAxes_1
            app.UIAxes_1 = uiaxes(app.SIMPLETab);
            xlabel(app.UIAxes_1, 'U ')
            ylabel(app.UIAxes_1, 'Y')
            zlabel(app.UIAxes_1, 'Z')
            app.UIAxes_1.Position = [8 51 204 165];

            % Create UIAxes_2
            app.UIAxes_2 = uiaxes(app.SIMPLETab);
            xlabel(app.UIAxes_2, 'U')
            ylabel(app.UIAxes_2, 'Y')
            zlabel(app.UIAxes_2, 'Z')
            app.UIAxes_2.Position = [201 49 212 165];

            % Create UIAxes_3
            app.UIAxes_3 = uiaxes(app.SIMPLETab);
            xlabel(app.UIAxes_3, 'U')
            ylabel(app.UIAxes_3, 'Y')
            zlabel(app.UIAxes_3, 'Z')
            app.UIAxes_3.Position = [403 49 224 165];

            % Create UIAxes_4
            app.UIAxes_4 = uiaxes(app.SIMPLETab);
            xlabel(app.UIAxes_4, 'U')
            ylabel(app.UIAxes_4, 'Y')
            zlabel(app.UIAxes_4, 'Z')
            app.UIAxes_4.Position = [622 49 213 165];

            % Create UIAxes_v
            app.UIAxes_v = uiaxes(app.SIMPLETab);
            title(app.UIAxes_v, 'Y-velocity Contours')
            xlabel(app.UIAxes_v, 'X')
            ylabel(app.UIAxes_v, 'Y')
            zlabel(app.UIAxes_v, 'Z')
            app.UIAxes_v.Position = [385 224 234 184];

            % Create UIAxes_overall
            app.UIAxes_overall = uiaxes(app.SIMPLETab);
            title(app.UIAxes_overall, 'Overall Velocity Contours')
            xlabel(app.UIAxes_overall, 'X')
            ylabel(app.UIAxes_overall, 'Y')
            zlabel(app.UIAxes_overall, 'Z')
            app.UIAxes_overall.Position = [617 224 238 183];

            % Create SIMPLEAlgorithmNavierStokesSolverforSteadyFlowLabel
            app.SIMPLEAlgorithmNavierStokesSolverforSteadyFlowLabel = uilabel(app.SIMPLETab);
            app.SIMPLEAlgorithmNavierStokesSolverforSteadyFlowLabel.FontSize = 18;
            app.SIMPLEAlgorithmNavierStokesSolverforSteadyFlowLabel.FontWeight = 'bold';
            app.SIMPLEAlgorithmNavierStokesSolverforSteadyFlowLabel.FontColor = [0 0 1];
            app.SIMPLEAlgorithmNavierStokesSolverforSteadyFlowLabel.Position = [23 514 711 44];
            app.SIMPLEAlgorithmNavierStokesSolverforSteadyFlowLabel.Text = 'SIMPLE Algorithm Navier Stokes Solver for Steady Flow Entering Parallel Plates ';

            % Create GenerateButton
            app.GenerateButton = uibutton(app.SIMPLETab, 'push');
            app.GenerateButton.ButtonPushedFcn = createCallbackFcn(app, @GenerateButtonPushed2, true);
            app.GenerateButton.BackgroundColor = [0.3882 0.8314 0.0706];
            app.GenerateButton.Position = [50 374 97 22];
            app.GenerateButton.Text = 'Generate';

            % Create InletVelocitymsEditFieldLabel
            app.InletVelocitymsEditFieldLabel = uilabel(app.SIMPLETab);
            app.InletVelocitymsEditFieldLabel.HorizontalAlignment = 'right';
            app.InletVelocitymsEditFieldLabel.Position = [67 488 103 22];
            app.InletVelocitymsEditFieldLabel.Text = 'Inlet Velocity (m/s)';

            % Create InletVelocitymsEditField
            app.InletVelocitymsEditField = uieditfield(app.SIMPLETab, 'numeric');
            app.InletVelocitymsEditField.ValueChangedFcn = createCallbackFcn(app, @InletVelocitymsEditFieldValueChanged, true);
            app.InletVelocitymsEditField.Position = [186 488 60 22];
            app.InletVelocitymsEditField.Value = 0.01;

            % Create UpperPlateVelocitymsEditField_2Label
            app.UpperPlateVelocitymsEditField_2Label = uilabel(app.SIMPLETab);
            app.UpperPlateVelocitymsEditField_2Label.HorizontalAlignment = 'right';
            app.UpperPlateVelocitymsEditField_2Label.Position = [35 453 147 22];
            app.UpperPlateVelocitymsEditField_2Label.Text = 'Upper Plate Velocity (m/s) ';

            % Create UpperPlateVelocitymsEditField_2
            app.UpperPlateVelocitymsEditField_2 = uieditfield(app.SIMPLETab, 'numeric');
            app.UpperPlateVelocitymsEditField_2.ValueChangedFcn = createCallbackFcn(app, @UpperPlateVelocitymsEditField_2ValueChanged, true);
            app.UpperPlateVelocitymsEditField_2.Position = [187 453 59 22];

            % Create LowerPlateVelocitymsEditField_2Label
            app.LowerPlateVelocitymsEditField_2Label = uilabel(app.SIMPLETab);
            app.LowerPlateVelocitymsEditField_2Label.HorizontalAlignment = 'right';
            app.LowerPlateVelocitymsEditField_2Label.Position = [26 419 144 22];
            app.LowerPlateVelocitymsEditField_2Label.Text = 'Lower Plate Velocity (m/s)';

            % Create LowerPlateVelocitymsEditField_2
            app.LowerPlateVelocitymsEditField_2 = uieditfield(app.SIMPLETab, 'numeric');
            app.LowerPlateVelocitymsEditField_2.ValueChangedFcn = createCallbackFcn(app, @LowerPlateVelocitymsEditField_2ValueChanged, true);
            app.LowerPlateVelocitymsEditField_2.Position = [187 419 59 22];

            % Create ProfileatxEditFieldLabel
            app.ProfileatxEditFieldLabel = uilabel(app.SIMPLETab);
            app.ProfileatxEditFieldLabel.HorizontalAlignment = 'right';
            app.ProfileatxEditFieldLabel.Position = [47 24 69 22];
            app.ProfileatxEditFieldLabel.Text = 'Profile at x=';

            % Create Profile_1
            app.Profile_1 = uieditfield(app.SIMPLETab, 'numeric');
            app.Profile_1.ValueChangedFcn = createCallbackFcn(app, @Profile_1ValueChanged2, true);
            app.Profile_1.Position = [124 24 65 22];

            % Create ProfileatxEditField_2Label
            app.ProfileatxEditField_2Label = uilabel(app.SIMPLETab);
            app.ProfileatxEditField_2Label.HorizontalAlignment = 'right';
            app.ProfileatxEditField_2Label.Position = [245 24 69 22];
            app.ProfileatxEditField_2Label.Text = 'Profile at x=';

            % Create Profile_2
            app.Profile_2 = uieditfield(app.SIMPLETab, 'numeric');
            app.Profile_2.ValueChangedFcn = createCallbackFcn(app, @Profile_2ValueChanged2, true);
            app.Profile_2.Position = [322 24 67 22];
            app.Profile_2.Value = 0.005;

            % Create ProfileatxEditField_3Label
            app.ProfileatxEditField_3Label = uilabel(app.SIMPLETab);
            app.ProfileatxEditField_3Label.HorizontalAlignment = 'right';
            app.ProfileatxEditField_3Label.Position = [451 24 69 22];
            app.ProfileatxEditField_3Label.Text = 'Profile at x=';

            % Create Profile_3
            app.Profile_3 = uieditfield(app.SIMPLETab, 'numeric');
            app.Profile_3.ValueChangedFcn = createCallbackFcn(app, @Profile_3ValueChanged2, true);
            app.Profile_3.Position = [528 24 68 22];
            app.Profile_3.Value = 0.02;

            % Create ProfileatxEditField_4Label
            app.ProfileatxEditField_4Label = uilabel(app.SIMPLETab);
            app.ProfileatxEditField_4Label.HorizontalAlignment = 'right';
            app.ProfileatxEditField_4Label.Position = [668 25 69 22];
            app.ProfileatxEditField_4Label.Text = 'Profile at x=';

            % Create Profile_4
            app.Profile_4 = uieditfield(app.SIMPLETab, 'numeric');
            app.Profile_4.ValueChangedFcn = createCallbackFcn(app, @Profile_4ValueChanged2, true);
            app.Profile_4.Position = [745 25 68 22];
            app.Profile_4.Value = 1;

            % Create Densitykgm3EditFieldLabel
            app.Densitykgm3EditFieldLabel = uilabel(app.SIMPLETab);
            app.Densitykgm3EditFieldLabel.HorizontalAlignment = 'right';
            app.Densitykgm3EditFieldLabel.Position = [353 488 95 22];
            app.Densitykgm3EditFieldLabel.Text = 'Density (kg/m^3)';

            % Create Densitykgm3EditField
            app.Densitykgm3EditField = uieditfield(app.SIMPLETab, 'numeric');
            app.Densitykgm3EditField.ValueChangedFcn = createCallbackFcn(app, @Densitykgm3EditFieldValueChanged, true);
            app.Densitykgm3EditField.Position = [464 488 65 22];
            app.Densitykgm3EditField.Value = 1000;

            % Create LengthmEditFieldLabel
            app.LengthmEditFieldLabel = uilabel(app.SIMPLETab);
            app.LengthmEditFieldLabel.HorizontalAlignment = 'right';
            app.LengthmEditFieldLabel.Position = [607 488 63 22];
            app.LengthmEditFieldLabel.Text = 'Length (m)';

            % Create LengthmEditField
            app.LengthmEditField = uieditfield(app.SIMPLETab, 'numeric');
            app.LengthmEditField.ValueChangedFcn = createCallbackFcn(app, @LengthmEditFieldValueChanged, true);
            app.LengthmEditField.Position = [686 488 41 22];
            app.LengthmEditField.Value = 1;

            % Create HeightmEditFieldLabel_2
            app.HeightmEditFieldLabel_2 = uilabel(app.SIMPLETab);
            app.HeightmEditFieldLabel_2.HorizontalAlignment = 'right';
            app.HeightmEditFieldLabel_2.Position = [387 423 61 22];
            app.HeightmEditFieldLabel_2.Text = 'Height (m)';

            % Create HeightmEditField
            app.HeightmEditField = uieditfield(app.SIMPLETab, 'numeric');
            app.HeightmEditField.ValueChangedFcn = createCallbackFcn(app, @HeightmEditFieldValueChanged, true);
            app.HeightmEditField.Position = [464 423 65 22];
            app.HeightmEditField.Value = 0.1;

            % Create DynamicViscositykgm1s1EditFieldLabel
            app.DynamicViscositykgm1s1EditFieldLabel = uilabel(app.SIMPLETab);
            app.DynamicViscositykgm1s1EditFieldLabel.HorizontalAlignment = 'right';
            app.DynamicViscositykgm1s1EditFieldLabel.Position = [278 456 170 22];
            app.DynamicViscositykgm1s1EditFieldLabel.Text = 'Dynamic Viscosity (kg m-1 s-1)';

            % Create DynamicViscositykgm1s1EditField
            app.DynamicViscositykgm1s1EditField = uieditfield(app.SIMPLETab, 'numeric');
            app.DynamicViscositykgm1s1EditField.ValueChangedFcn = createCallbackFcn(app, @DynamicViscositykgm1s1EditFieldValueChanged, true);
            app.DynamicViscositykgm1s1EditField.Position = [464 456 65 22];
            app.DynamicViscositykgm1s1EditField.Value = 0.001;

            % Create updated
            app.updated = uibutton(app.SIMPLETab, 'push');
            app.updated.BackgroundColor = [1 0 0];
            app.updated.Position = [21 374 24 22];
            app.updated.Text = '';

            % Create maxvelocityinthexdirectionEditFieldLabel
            app.maxvelocityinthexdirectionEditFieldLabel = uilabel(app.SIMPLETab);
            app.maxvelocityinthexdirectionEditFieldLabel.HorizontalAlignment = 'right';
            app.maxvelocityinthexdirectionEditFieldLabel.Position = [2 291 162 22];
            app.maxvelocityinthexdirectionEditFieldLabel.Text = 'max velocity in the x-direction';

            % Create maxVelSIMPLE
            app.maxVelSIMPLE = uieditfield(app.SIMPLETab, 'numeric');
            app.maxVelSIMPLE.BackgroundColor = [0.902 0.902 0.902];
            app.maxVelSIMPLE.Position = [56 269 63 22];

            % Create pressuregradientforUinletLabel
            app.pressuregradientforUinletLabel = uilabel(app.SIMPLETab);
            app.pressuregradientforUinletLabel.BackgroundColor = [1 1 1];
            app.pressuregradientforUinletLabel.HorizontalAlignment = 'right';
            app.pressuregradientforUinletLabel.Position = [22 242 128 22];
            app.pressuregradientforUinletLabel.Text = 'x-dir pressure gradient ';

            % Create Gsimple
            app.Gsimple = uieditfield(app.SIMPLETab, 'numeric');
            app.Gsimple.BackgroundColor = [0.902 0.902 0.902];
            app.Gsimple.Position = [56 220 63 22];

            % Create exportWorkspace
            app.exportWorkspace = uibutton(app.SIMPLETab, 'push');
            app.exportWorkspace.ButtonPushedFcn = createCallbackFcn(app, @exportWorkspaceButtonPushed, true);
            app.exportWorkspace.Position = [19 344 128 23];
            app.exportWorkspace.Text = 'export to workspace';

            % Create ReEditFieldLabel
            app.ReEditFieldLabel = uilabel(app.SIMPLETab);
            app.ReEditFieldLabel.BackgroundColor = [1 1 1];
            app.ReEditFieldLabel.HorizontalAlignment = 'right';
            app.ReEditFieldLabel.FontColor = [0.1098 0.6784 0.102];
            app.ReEditFieldLabel.Position = [738 524 25 22];
            app.ReEditFieldLabel.Text = 'Re';

            % Create PressureContourSwitchLabel
            app.PressureContourSwitchLabel = uilabel(app.SIMPLETab);
            app.PressureContourSwitchLabel.HorizontalAlignment = 'center';
            app.PressureContourSwitchLabel.Position = [704 456 99 22];
            app.PressureContourSwitchLabel.Text = 'Pressure Contour';

            % Create PressureContourSwitch
            app.PressureContourSwitch = uiswitch(app.SIMPLETab, 'slider');
            app.PressureContourSwitch.Position = [638 459 36 16];

            % Create VelocityQuiverPlotSwitchLabel
            app.VelocityQuiverPlotSwitchLabel = uilabel(app.SIMPLETab);
            app.VelocityQuiverPlotSwitchLabel.HorizontalAlignment = 'center';
            app.VelocityQuiverPlotSwitchLabel.Position = [699 424 109 22];
            app.VelocityQuiverPlotSwitchLabel.Text = 'Velocity Quiver Plot';

            % Create VelocityQuiverPlotSwitch
            app.VelocityQuiverPlotSwitch = uiswitch(app.SIMPLETab, 'slider');
            app.VelocityQuiverPlotSwitch.Position = [638 427 36 16];

            % Create ReEditField
            app.ReEditField = uieditfield(app.SIMPLETab, 'numeric');
            app.ReEditField.ValueChangedFcn = createCallbackFcn(app, @ReEditFieldValueChanged, true);
            app.ReEditField.FontColor = [0.1098 0.6784 0.102];
            app.ReEditField.BackgroundColor = [0.902 0.902 0.902];
            app.ReEditField.Position = [773 524 55 22];

            % Create AnalyticalTab
            app.AnalyticalTab = uitab(app.TabGroup);
            app.AnalyticalTab.Title = 'Analytical';
            app.AnalyticalTab.BackgroundColor = [1 1 1];

            % Create UIAxes1
            app.UIAxes1 = uiaxes(app.AnalyticalTab);
            title(app.UIAxes1, 'Velocity Profile')
            xlabel(app.UIAxes1, 'X')
            ylabel(app.UIAxes1, 'Y')
            zlabel(app.UIAxes1, 'Z')
            app.UIAxes1.Position = [214 224 300 185];

            % Create UIAxes3
            app.UIAxes3 = uiaxes(app.AnalyticalTab);
            title(app.UIAxes3, 'Stress Profile')
            xlabel(app.UIAxes3, 'X')
            ylabel(app.UIAxes3, 'Y')
            zlabel(app.UIAxes3, 'Z')
            app.UIAxes3.Position = [217 32 300 185];

            % Create UIAxes2
            app.UIAxes2 = uiaxes(app.AnalyticalTab);
            title(app.UIAxes2, 'Velocity Contour')
            xlabel(app.UIAxes2, 'X')
            ylabel(app.UIAxes2, 'Y')
            zlabel(app.UIAxes2, 'Z')
            app.UIAxes2.Position = [561 223 300 185];

            % Create UIAxes4
            app.UIAxes4 = uiaxes(app.AnalyticalTab);
            title(app.UIAxes4, 'Stress Contour')
            xlabel(app.UIAxes4, 'X')
            ylabel(app.UIAxes4, 'Y')
            zlabel(app.UIAxes4, 'Z')
            app.UIAxes4.Position = [561 32 300 185];

            % Create Label
            app.Label = uilabel(app.AnalyticalTab);
            app.Label.FontSize = 18;
            app.Label.FontWeight = 'bold';
            app.Label.FontColor = [0 0 1];
            app.Label.Position = [21 524 655 23];
            app.Label.Text = 'Fully Developed Flow Between Parallel Plates, Stress and Velocity Profiles';

            % Create HeightmEditFieldLabel
            app.HeightmEditFieldLabel = uilabel(app.AnalyticalTab);
            app.HeightmEditFieldLabel.BackgroundColor = [1 1 1];
            app.HeightmEditFieldLabel.HorizontalAlignment = 'right';
            app.HeightmEditFieldLabel.Position = [142 459 61 22];
            app.HeightmEditFieldLabel.Text = 'Height (m)';

            % Create HeightEditField
            app.HeightEditField = uieditfield(app.AnalyticalTab, 'numeric');
            app.HeightEditField.ValueChangedFcn = createCallbackFcn(app, @HeightEditFieldValueChanged, true);
            app.HeightEditField.Position = [220 459 68 22];
            app.HeightEditField.Value = 0.1;

            % Create DynamicViscosityPasEditFieldLabel
            app.DynamicViscosityPasEditFieldLabel = uilabel(app.AnalyticalTab);
            app.DynamicViscosityPasEditFieldLabel.HorizontalAlignment = 'right';
            app.DynamicViscosityPasEditFieldLabel.Position = [66 423 137 22];
            app.DynamicViscosityPasEditFieldLabel.Text = 'Dynamic Viscosity (Pa s)';

            % Create DynamicViscosityEditField
            app.DynamicViscosityEditField = uieditfield(app.AnalyticalTab, 'numeric');
            app.DynamicViscosityEditField.ValueChangedFcn = createCallbackFcn(app, @DynamicViscosityEditFieldValueChanged, true);
            app.DynamicViscosityEditField.Position = [220 423 68 22];
            app.DynamicViscosityEditField.Value = 0.001;

            % Create UpperPlateVelocitymsEditFieldLabel
            app.UpperPlateVelocitymsEditFieldLabel = uilabel(app.AnalyticalTab);
            app.UpperPlateVelocitymsEditFieldLabel.HorizontalAlignment = 'right';
            app.UpperPlateVelocitymsEditFieldLabel.Position = [320 458 147 22];
            app.UpperPlateVelocitymsEditFieldLabel.Text = 'Upper Plate Velocity (m/s) ';

            % Create UpperPlateVelocitymsEditField
            app.UpperPlateVelocitymsEditField = uieditfield(app.AnalyticalTab, 'numeric');
            app.UpperPlateVelocitymsEditField.ValueChangedFcn = createCallbackFcn(app, @UpperPlateVelocitymsEditFieldValueChanged, true);
            app.UpperPlateVelocitymsEditField.Position = [472 458 39 22];

            % Create LowerPlateVelocitymsEditFieldLabel
            app.LowerPlateVelocitymsEditFieldLabel = uilabel(app.AnalyticalTab);
            app.LowerPlateVelocitymsEditFieldLabel.HorizontalAlignment = 'right';
            app.LowerPlateVelocitymsEditFieldLabel.Position = [321 424 144 22];
            app.LowerPlateVelocitymsEditFieldLabel.Text = 'Lower Plate Velocity (m/s)';

            % Create LowerPlateVelocitymsEditField
            app.LowerPlateVelocitymsEditField = uieditfield(app.AnalyticalTab, 'numeric');
            app.LowerPlateVelocitymsEditField.ValueChangedFcn = createCallbackFcn(app, @LowerPlateVelocitymsEditFieldValueChanged, true);
            app.LowerPlateVelocitymsEditField.Position = [472 424 40 22];

            % Create XdirPressureGradientPamEditFieldLabel
            app.XdirPressureGradientPamEditFieldLabel = uilabel(app.AnalyticalTab);
            app.XdirPressureGradientPamEditFieldLabel.HorizontalAlignment = 'right';
            app.XdirPressureGradientPamEditFieldLabel.Position = [548 459 170 22];
            app.XdirPressureGradientPamEditFieldLabel.Text = 'X-dir Pressure Gradient (Pa/m)';

            % Create XdirPressureGradientPamEditField
            app.XdirPressureGradientPamEditField = uieditfield(app.AnalyticalTab, 'numeric');
            app.XdirPressureGradientPamEditField.ValueChangedFcn = createCallbackFcn(app, @XdirPressureGradientPamEditFieldValueChanged, true);
            app.XdirPressureGradientPamEditField.Position = [735 459 89 22];
            app.XdirPressureGradientPamEditField.Value = -0.012;

            % Create GenerateButton_2
            app.GenerateButton_2 = uibutton(app.AnalyticalTab, 'push');
            app.GenerateButton_2.ButtonPushedFcn = createCallbackFcn(app, @GenerateButton_2Pushed, true);
            app.GenerateButton_2.BackgroundColor = [0.3922 0.8314 0.0745];
            app.GenerateButton_2.Position = [65 373 100 24];
            app.GenerateButton_2.Text = 'Generate';

            % Create MaxVelocitymsLabel
            app.MaxVelocitymsLabel = uilabel(app.AnalyticalTab);
            app.MaxVelocitymsLabel.BackgroundColor = [1 1 1];
            app.MaxVelocitymsLabel.HorizontalAlignment = 'right';
            app.MaxVelocitymsLabel.Position = [23 281 106 22];
            app.MaxVelocitymsLabel.Text = 'Max Velocity (m/s):';

            % Create MaxVelocitymsEditField
            app.MaxVelocitymsEditField = uieditfield(app.AnalyticalTab, 'numeric');
            app.MaxVelocitymsEditField.BackgroundColor = [0.9412 0.9412 0.9412];
            app.MaxVelocitymsEditField.Position = [137 281 59 22];

            % Create MaxVelocitymsEditField_2Label_2
            app.MaxVelocitymsEditField_2Label_2 = uilabel(app.AnalyticalTab);
            app.MaxVelocitymsEditField_2Label_2.BackgroundColor = [1 1 1];
            app.MaxVelocitymsEditField_2Label_2.HorizontalAlignment = 'right';
            app.MaxVelocitymsEditField_2Label_2.Position = [8 253 123 22];
            app.MaxVelocitymsEditField_2Label_2.Text = 'Y-Position of max (m):';

            % Create YPositionofmaxmEditField
            app.YPositionofmaxmEditField = uieditfield(app.AnalyticalTab, 'numeric');
            app.YPositionofmaxmEditField.BackgroundColor = [0.9412 0.9412 0.9412];
            app.YPositionofmaxmEditField.Position = [137 253 59 22];

            % Create MinVelocitymsEditFieldLabel
            app.MinVelocitymsEditFieldLabel = uilabel(app.AnalyticalTab);
            app.MinVelocitymsEditFieldLabel.BackgroundColor = [1 1 1];
            app.MinVelocitymsEditFieldLabel.HorizontalAlignment = 'right';
            app.MinVelocitymsEditFieldLabel.Position = [25 219 103 22];
            app.MinVelocitymsEditFieldLabel.Text = 'Min Velocity (m/s):';

            % Create MinVelocitymsEditField
            app.MinVelocitymsEditField = uieditfield(app.AnalyticalTab, 'numeric');
            app.MinVelocitymsEditField.BackgroundColor = [0.9412 0.9412 0.9412];
            app.MinVelocitymsEditField.Position = [136 219 59 22];

            % Create YPositionofminmEditFieldLabel
            app.YPositionofminmEditFieldLabel = uilabel(app.AnalyticalTab);
            app.YPositionofminmEditFieldLabel.BackgroundColor = [1 1 1];
            app.YPositionofminmEditFieldLabel.HorizontalAlignment = 'right';
            app.YPositionofminmEditFieldLabel.Position = [11 191 119 22];
            app.YPositionofminmEditFieldLabel.Text = 'Y-Position of min (m):';

            % Create YPositionofminmEditField
            app.YPositionofminmEditField = uieditfield(app.AnalyticalTab, 'numeric');
            app.YPositionofminmEditField.BackgroundColor = [0.9412 0.9412 0.9412];
            app.YPositionofminmEditField.Position = [136 191 59 22];

            % Create Label_6
            app.Label_6 = uilabel(app.AnalyticalTab);
            app.Label_6.WordWrap = 'on';
            app.Label_6.Position = [22 494 595 31];
            app.Label_6.Text = 'Analytical solutions and profiles for fully developed flow problems, given parameter values';

            % Create KeyResultsLabel
            app.KeyResultsLabel = uilabel(app.AnalyticalTab);
            app.KeyResultsLabel.HorizontalAlignment = 'center';
            app.KeyResultsLabel.FontSize = 14;
            app.KeyResultsLabel.FontWeight = 'bold';
            app.KeyResultsLabel.Position = [11 312 184 22];
            app.KeyResultsLabel.Text = 'Key Results';

            % Create updated_2
            app.updated_2 = uibutton(app.AnalyticalTab, 'push');
            app.updated_2.BackgroundColor = [1 0 0];
            app.updated_2.Position = [39 373 22 23];
            app.updated_2.Text = '';

            % Create exportAnalytical
            app.exportAnalytical = uibutton(app.AnalyticalTab, 'push');
            app.exportAnalytical.ButtonPushedFcn = createCallbackFcn(app, @exportAnalyticalButtonPushed, true);
            app.exportAnalytical.Position = [40 343 124 23];
            app.exportAnalytical.Text = 'export to workspace';

            % Create AverageVelocityEditFieldLabel
            app.AverageVelocityEditFieldLabel = uilabel(app.AnalyticalTab);
            app.AverageVelocityEditFieldLabel.BackgroundColor = [1 1 1];
            app.AverageVelocityEditFieldLabel.HorizontalAlignment = 'right';
            app.AverageVelocityEditFieldLabel.Position = [22 124 98 22];
            app.AverageVelocityEditFieldLabel.Text = 'Average Velocity:';

            % Create AverageVelocityEditField
            app.AverageVelocityEditField = uieditfield(app.AnalyticalTab, 'numeric');
            app.AverageVelocityEditField.BackgroundColor = [0.9412 0.9412 0.9412];
            app.AverageVelocityEditField.Position = [134 124 61 22];

            % Create ReEditField_2Label
            app.ReEditField_2Label = uilabel(app.AnalyticalTab);
            app.ReEditField_2Label.BackgroundColor = [1 1 1];
            app.ReEditField_2Label.HorizontalAlignment = 'right';
            app.ReEditField_2Label.FontColor = [0.1098 0.6784 0.102];
            app.ReEditField_2Label.Position = [70 91 25 22];
            app.ReEditField_2Label.Text = 'Re';

            % Create ReEditField_Ana
            app.ReEditField_Ana = uieditfield(app.AnalyticalTab, 'numeric');
            app.ReEditField_Ana.FontColor = [0.1098 0.6784 0.102];
            app.ReEditField_Ana.BackgroundColor = [0.9412 0.9412 0.9412];
            app.ReEditField_Ana.Position = [105 91 55 22];
            app.ReEditField_Ana.Value = 1000;

            % Create SettingsTab
            app.SettingsTab = uitab(app.TabGroup);
            app.SettingsTab.Title = 'Settings';
            app.SettingsTab.BackgroundColor = [0.902 0.902 0.902];

            % Create gridpointsinydirectionEditFieldLabel
            app.gridpointsinydirectionEditFieldLabel = uilabel(app.SettingsTab);
            app.gridpointsinydirectionEditFieldLabel.HorizontalAlignment = 'right';
            app.gridpointsinydirectionEditFieldLabel.Position = [22 503 132 22];
            app.gridpointsinydirectionEditFieldLabel.Text = 'grid points in y-direction';

            % Create gridpointsinydirectionEditField
            app.gridpointsinydirectionEditField = uieditfield(app.SettingsTab, 'numeric');
            app.gridpointsinydirectionEditField.Position = [169 503 100 22];
            app.gridpointsinydirectionEditField.Value = 30;

            % Create StreamlinesoncontourplotsSwitchLabel
            app.StreamlinesoncontourplotsSwitchLabel = uilabel(app.SettingsTab);
            app.StreamlinesoncontourplotsSwitchLabel.HorizontalAlignment = 'center';
            app.StreamlinesoncontourplotsSwitchLabel.Position = [23 471 157 22];
            app.StreamlinesoncontourplotsSwitchLabel.Text = 'Streamlines on contour plots';

            % Create StreamlinesOnOff
            app.StreamlinesOnOff = uiswitch(app.SettingsTab, 'slider');
            app.StreamlinesOnOff.Items = {'On', 'Off'};
            app.StreamlinesOnOff.Position = [210 472 45 20];
            app.StreamlinesOnOff.Value = 'On';

            % Create SIMPLETabLabel
            app.SIMPLETabLabel = uilabel(app.SettingsTab);
            app.SIMPLETabLabel.FontSize = 16;
            app.SIMPLETabLabel.FontWeight = 'bold';
            app.SIMPLETabLabel.Position = [22 524 101 22];
            app.SIMPLETabLabel.Text = 'SIMPLE Tab ';

            % Create AnalyticalTabLabel
            app.AnalyticalTabLabel = uilabel(app.SettingsTab);
            app.AnalyticalTabLabel.FontSize = 16;
            app.AnalyticalTabLabel.FontWeight = 'bold';
            app.AnalyticalTabLabel.Position = [22 414 112 22];
            app.AnalyticalTabLabel.Text = 'Analytical Tab';

            % Create gridpointsinydirectionEditField_2Label
            app.gridpointsinydirectionEditField_2Label = uilabel(app.SettingsTab);
            app.gridpointsinydirectionEditField_2Label.HorizontalAlignment = 'right';
            app.gridpointsinydirectionEditField_2Label.Position = [19 393 132 22];
            app.gridpointsinydirectionEditField_2Label.Text = 'grid points in y-direction';

            % Create gridPtsAnalytical
            app.gridPtsAnalytical = uieditfield(app.SettingsTab, 'numeric');
            app.gridPtsAnalytical.ValueChangedFcn = createCallbackFcn(app, @gridPtsAnalyticalValueChanged, true);
            app.gridPtsAnalytical.Position = [169 393 100 22];
            app.gridPtsAnalytical.Value = 150;

            % Create RESETALLButton
            app.RESETALLButton = uibutton(app.SettingsTab, 'push');
            app.RESETALLButton.ButtonPushedFcn = createCallbackFcn(app, @RESETALLButtonPushed, true);
            app.RESETALLButton.BackgroundColor = [1 0 0];
            app.RESETALLButton.FontWeight = 'bold';
            app.RESETALLButton.FontColor = [1 1 1];
            app.RESETALLButton.Position = [20 323 100 23];
            app.RESETALLButton.Text = 'RESET ALL';

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = Copy_of_MATSCF_noDImTab_exported

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            % Execute the startup function
            runStartupFcn(app, @startupFcn)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end
