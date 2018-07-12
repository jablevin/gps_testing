import os
import numpy as np
import matplotlib.pyplot as plt
import trajan as tr
import windlass as wl
from gmplot import gmplot
from scipy.interpolate import interp1d
import time
import pandas
import scipy.optimize as optimization
from numba import jit
import math

path = os.path.expanduser(os.path.join('~', 'temp-drive', 'ProcessedData'))
plot = True
unc = True
lidar_active = False

colors = {'lidar':'darkorange', 'imu':'crimson', 'gps':'lime', 'ins':'blue'}

for run_name in [#'20180529_3433Outside_RadDet_Run009']:
        # '20180529_3433Outside_RadDet_Run009',
        # '20180529_3433Outside_RadDet_Run008',
        # '20180529_3433Outside_RadDet_Run007',
        # '20180529_3433Outside_RadDet_Run006',
        # '20180529_3433Outside_RadDet_Run005',
        # '20180529_3433Outside_RadDet_Run004',
        # '20180529_3433Outside_RadDet_Run002',
        # '20180529_3433Outside_RadDet_Run001',
        # '20180606_70parkinglot_run01_south',
        # '20180606_70parkinglot_run02_north',
        # '20180606_70parkinglot_run03_to_loading_dock',
        # '20180606_70parkinglot_run04_loading_dock_to_50c',
        # '20180611_50Cto70laodingdock_Run001',
        # '20180611_70loadingdockto50C_Run002',
        # '20180626_run01_antenna_shielded_elevated',
        '20180626_run02_antenna_shielded_elevated',
]:

    start = time.time()
    run_name = '20180626_run01_antenna_shielded_elevated'
    path_run = os.path.join(path, run_name)

    if run_name.startswith('20180529'):
        name = run_name[-6:]
    if run_name.startswith('20180606'):
        name = run_name[-11:]
    if run_name.startswith('20180606') and run_name.endswith('dock'):
        name = run_name[-21:]
    if run_name.startswith('20180611') or run_name.endswith('50c'):
        name = run_name[-25:]
    if run_name.startswith('20180626'):
        name = run_name[-31:]

    if not unc:
        imu = np.loadtxt(os.path.join(path_run, 'vectornav_IMU.csv'), skiprows = 1, delimiter = ',')
        gps = np.loadtxt(os.path.join(path_run, 'vectornav_GPS.csv'), skiprows = 1, delimiter = ',')
        if not os.path.exists('/home/jacob/DNDO/IMU_GPS/' + name):
            string = 'mkdir /home/jacob/DNDO/IMU_GPS/' + name
            os.system(string)
        vec_data = {'t': gps[:,0], 'alt': gps[:,1], 'lat': gps[:,2] , 'long': gps[:,3]}
        ins_time = [vec_data['t'][i] - vec_data['t'][0] for i in range(len(vec_data['t']))]

    if unc == True:
        path = os.path.expanduser(os.path.join('~', 'temp-drive', 'ProcessedData'))#, 'Extracted'))
        path_run = os.path.join(path, run_name)
        if not os.path.exists('/home/jacob/DNDO/IMU_GPS/' + name):
            string = 'mkdir /home/jacob/DNDO/IMU_GPS/' + name
            os.system(string)
        pd_gps = pandas.read_csv(os.path.join(path_run, 'vectornav_gps.csv'))
        pd_ins = pandas.read_csv(os.path.join(path_run, 'vectornav_ins.csv'))
        pd_imu = pandas.read_csv(os.path.join(path_run, 'vectornav_imu.csv'))
        ins_time = [i - pd_ins['timestamp_unix'][0] for i in pd_ins['timestamp_unix']]
        gps_time = [i - pd_ins['timestamp_unix'][0] for i in pd_gps['timestamp_unix']]

    if plot:
        gmap = gmplot.GoogleMapPlotter(pd_ins['Latitude'][0], pd_ins['Longitude'][0], 18)
        gmap.plot(pd_ins['Latitude'], pd_ins['Longitude'], colors['ins'], edge_width=7)
        gmap.plot(pd_gps['Latitude'], pd_gps['Longitude'], colors['gps'], edge_width=7)
        gmap.scatter([pd_ins['Latitude'][0]], [pd_ins['Longitude'][0]], 'green')
        gmap.scatter([pd_ins['Latitude'].iloc[-1]], [pd_ins['Longitude'].iloc[-1]], 'red')
        gmap.scatter([pd_gps['Latitude'][0]], [pd_gps['Longitude'][0]], 'green')
        gmap.scatter([pd_gps['Latitude'].iloc[-1]], [pd_gps['Longitude'].iloc[-1]], 'red')
        gmap.draw('/home/jacob/DNDO/IMU_GPS/' + name + '/ins_gps.html')
    # input('break')
    #########################################################################################################################################################################################################
    #  Testing Heading of IMU for Trajectory
    #########################################################################################################################################################################################################
    def savitzky_golay(y, window_size, order, deriv=0, rate=1):
            from math import factorial
            import numpy as np
            try:
                window_size = np.abs(np.int(window_size))
                order = np.abs(np.int(order))
            except ValueError:
                raise ValueError("window_size and order have to be of type int")
            if window_size % 2 != 1 or window_size < 1:
                raise TypeError("window_size size must be a positive odd number")
            if window_size < order + 2:
                raise TypeError("window_size is too small for the polynomials order")
            order_range = range(order+1)
            half_window = (window_size -1) // 2
            b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
            m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
            firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
            lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
            y = np.concatenate((firstvals, y, lastvals))
            return np.convolve( m[::-1], y, mode='valid')

    def calc_compass_bearing(pointA, pointB):
        if (type(pointA) != tuple) or (type(pointB) != tuple):
            raise TypeError("Only tuples are supported as arguments")
        lat1 = np.radians(pointA[0])
        lat2 = np.radians(pointB[0])
        diffLong = np.radians(pointB[1] - pointA[1])
        x = np.sin(diffLong) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1)
                * np.cos(lat2) * np.cos(diffLong))
        initial_bearing = math.atan2(x, y)
        initial_bearing = np.degrees(initial_bearing)
        compass_bearing = (initial_bearing + 360) % 360
        return compass_bearing

    imu_time = [i - pd_imu['timestamp_unix'][0] for i in pd_imu['timestamp_unix']]
    ax = np.array(pd_imu['Accel_X'])
    ay = np.array(pd_imu['Accel_Y'])
    az = np.array(pd_imu['Accel_Z'])

    if plot:
        plt.figure(figsize = (12,8))
        plt.plot(imu_time, ax, label = 'Acc. in X')
        plt.plot(imu_time, ay, label = 'Acc. in Y')
        plt.plot(imu_time, az, label = 'Acc. in Z')
        plt.xlabel('Time [s]')
        plt.ylabel(r'Acceleration [$m/s^2$]')
        plt.legend()
        plt.tight_layout()
        plt.savefig('/home/jacob/DNDO/IMU_GPS/' + name + '/unfiltered_acc')
        plt.close()
        # plt.show()

    ax = savitzky_golay(ax, 51, 3)
    ay = savitzky_golay(ay, 51, 3)
    az = savitzky_golay(az, 51, 3)
    roll = np.array([i for i in pd_ins['Roll']])
    pitch = np.array([i * -1 for i in pd_ins['Pitch']])
    yaw = np.array([i * -1 for i in pd_ins['Yaw']])
    # yaw[0]
    # plt.figure()
    # plt.plot(ins_time, yaw)
    # plt.show()

    for i in range(len(roll)):
        if np.sign(roll[i]) == -1:
            roll[i] = roll[i] + 180
        else:
            roll[i] = roll[i] - 180

    acc_mag = [np.sqrt(ax[i]**2 + ay[i]**2 + az[i]**2) for i in range(len(ax))]

    if plot:
        plt.figure(figsize = (12,8))
        plt.plot(imu_time, acc_mag, color = 'red', label = 'Acceleration Magnitude')
        plt.xlabel('Time [s]')
        plt.ylabel(r'Acceleration [$m/s^2$]')
        plt.legend()
        plt.tight_layout()
        plt.savefig('/home/jacob/DNDO/IMU_GPS/' + name + '/filtered_acc_magnitude')
        plt.close()
        # plt.show()
        np.mean(acc_mag[50:200])
    gravity = np.mean(acc_mag[50:200])

    f_ax = interp1d(imu_time, ax, axis = 0, fill_value = 'extrapolate')
    f_ay = interp1d(imu_time, ay, axis = 0, fill_value = 'extrapolate')
    f_az = interp1d(imu_time, az, axis = 0, fill_value = 'extrapolate')

    new_ax = f_ax(ins_time)
    new_ay = f_ay(ins_time)
    new_az = f_az(ins_time)

    corrected_ax = [new_ax[i] + new_az[i] * np.sin(np.radians(pitch[i]))
            + new_ay[i] * np.sin(np.radians(roll[i])) for i in range(len(pitch))]

    corrected_ay = [new_ay[i] - new_az[i] * np.sin(np.radians(roll[i])) for i in range(len(pitch))]

    corrected_az = [new_az[i] - new_ax[i] * np.sin(np.radians(pitch[i]))
            - new_ay[i] * np.sin(np.radians(roll[i])) for i in range(len(pitch))]

    # vel_x = [np.trapz(corrected_ax[0:i], ins_time[0:i]) for i in range(len(ins_time))]
    # pos_x = [np.trapz(vel_x[0:i], ins_time[0:i]) for i in range(len(ins_time))]
    # vel_y = [np.trapz(corrected_ay[0:i], ins_time[0:i]) for i in range(len(ins_time))]
    # pos_y = [np.trapz(vel_y[0:i], ins_time[0:i]) for i in range(len(ins_time))]

    if plot:
        # plt.figure(figsize = (12,8))
        # plt.plot(ins_time, vel_x, label = 'Vel. X')
        # plt.plot(ins_time, vel_y, label = 'Vel. Y')
        # plt.xlabel('Time [s]')
        # plt.ylabel(r'Velocity [$m/s$]')
        # plt.tight_layout()
        # plt.savefig('/home/jacob/DNDO/IMU_GPS/' + name + '/imu_velocity')
        # plt.close()
        #
        # plt.figure(figsize = (12,8))
        # plt.plot(pos_x, pos_y)
        # plt.xlabel('Pos. X [m]')
        # plt.ylabel('Pos. Y [m]')
        # plt.tight_layout()
        # plt.savefig('/home/jacob/DNDO/IMU_GPS/' + name + '/imu_position_estimate')
        # plt.close()

        plt.figure(figsize = (12,8))
        plt.plot(ins_time, roll, label = 'Roll')
        plt.plot(ins_time, pitch, label = 'Pitch')
        plt.plot(ins_time, yaw, label = 'Yaw')
        plt.xlabel('Time [s]')
        plt.ylabel('Angles [Degrees]')
        plt.legend()
        plt.tight_layout()
        plt.savefig('/home/jacob/DNDO/IMU_GPS/' + name + '/roll_pitch_yaw')
        plt.close()
        # plt.show()

    ins_lat = tuple(pd_ins['Latitude'])
    ins_long = tuple(pd_ins['Longitude'])
    points = [[ins_lat[i], ins_long[i]] for i in range(len(ins_time))]

    bearing = [calc_compass_bearing(tuple(points[i]) , tuple(points[i+1])) for i in range(len(ins_time)-1)]
    new_bearing = savitzky_golay(bearing, 51, 3)
    # break
    #########################################################################################################################################################################################################
    #  Comparing lidar trajectory versus that of vectornav
    #########################################################################################################################################################################################################

    @jit
    def rotate(origin, point, angle):
        ox, oy = origin
        px, py = point
        qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return qx, qy

    @jit
    def gps_to_xy(lat, long):
        locations = np.zeros((len(lat), 2))
        for i in range(len(lat)-1):
            locations[i+1,0] = locations[i,0] + ((long[i+1] - long[i]) * np.pi * 6371 * 1000 / 180 * np.cos(lat[i+1] * np.pi / 180))
            locations[i+1,1] = locations[i,1] + ((lat[i+1] - lat[i]) * np.pi * 6371 * 1000 / 180)
        return locations

    def traj_fit(params, lidar_loc, resampled_loc, sat_bool, sigma_g):
        x, y, theta = params
        rotated = np.zeros((len(lidar_loc), 2))
        for i in range(len(lidar_loc)):
            rotated[i] = rotate([0,0], lidar_loc[i], np.radians(theta))
        ct = [i + [x, y] for i in rotated]
        result = 0
        for i in range(len(resampled_loc)):
            R = np.linalg.norm(ct[i] - resampled_loc[i]) * sat_bool[i] / np.sqrt(.5**2 + sigma_g[i]**2)
            result += R**2
        return result

    def yaw_fit(offset, imu_yaw, ins_yaw, uncertainty, sat_bool):
        holder = np.zeros(len(imu_yaw))
        for i in range(len(imu_yaw)):
            holder[i] = imu_yaw[i] + offset
        result = 0
        for i in range(len(imu_yaw)):
            R = (ins_yaw[i] - holder[i]) / (uncertainty[i])
            result += R**2
        return result

    flag1 = time.time()
    new_t = np.linspace(0, ins_time[-1], int(ins_time[-1])+1)

    if lidar_active:
        traj1 = tr.Trajectory.from_file(os.path.join(path_run, 'pose_graph.pbstream')).resample(1)
        pc_traj = wl.utilities.make_trajectory(traj1, length=0.5)
        new_t = [i - traj1.t[0] for i in traj1.t]
        conv_rpy = []
        for i in range(len(traj1.qw)):
            new_obj = tr.Rotation([traj1.qx[i], traj1.qy[i], traj1.qz[i],traj1.qw[i]])
            conv_rpy.append(new_obj.to_rpy())
        lidar_angles = np.array(conv_rpy)
        lidar_loc = np.zeros((len(new_t), 2))
        for i in range(len(new_t)):
            lidar_loc[i] = [traj1.px[i], traj1.py[i]]

    flag2 = time.time()

    if not unc:
        lat = [i for i in vec_data['lat']]
        long = [i for i in vec_data['long']]
    if unc:
        lat = [i for i in pd_ins['Latitude']]
        long = [i for i in pd_ins['Longitude']]

    vec_locations = gps_to_xy(lat, long)

    fx = interp1d(ins_time, vec_locations[:,0], axis = 0, fill_value = 'extrapolate')
    fy = interp1d(ins_time, vec_locations[:,1], axis = 0, fill_value = 'extrapolate')
    f_fix = interp1d(gps_time, pd_gps['GpsFix'], axis = 0, fill_value = 'extrapolate')
    f_sigma = interp1d(ins_time, pd_ins['Pos_Unc'], axis = 0, fill_value = 'extrapolate')
    f_lat = interp1d(ins_time, pd_ins['Latitude'], axis = 0, fill_value = 'extrapolate')
    f_long = interp1d(ins_time, pd_ins['Longitude'], axis = 0, fill_value = 'extrapolate')
    f_numsat = interp1d(gps_time, pd_gps['NumSats'], axis = 0, fill_value = 'extrapolate')
    f_bearing = interp1d(ins_time[:-1], new_bearing, axis = 0, fill_value = 'extrapolate')
    f_yaw = interp1d(ins_time, yaw, axis = 0, fill_value = 'extrapolate')

    N = 1
    yaw_t = np.linspace(0, new_t[-1], len(new_t) // N)

    new_x = fx(new_t)
    new_y = fy(new_t)
    gps_fix = f_fix(imu_time)
    new_lat = f_lat(new_t)
    new_long = f_long(new_t)
    num_sats = f_numsat(new_t)
    res_bearing = f_bearing(yaw_t)
    res_yaw = f_yaw(yaw_t)

    resampled_loc = np.zeros((len(new_t), 2))
    for i in range(len(new_t)):
        resampled_loc[i] = [new_x[i], new_y[i]]

    arrows = np.zeros((len(new_t), 4))
    dist = 3
    for i in range(len(yaw_t)):
        arrows[i,0] = resampled_loc[N*i,0]
        arrows[i,1] = resampled_loc[N*i,1]
        arrows[i,2] = dist * -np.sin(np.radians(res_yaw[i]))
        arrows[i,3] = dist * np.cos(np.radians(res_yaw[i]))

    temp_yaw = -res_yaw
    for i in range(len(res_yaw)):
        if np.sign(temp_yaw[i]) == -1:
            temp_yaw[i] += 360

    if not lidar_active:
        num_sats = f_numsat(yaw_t)
        ins_unc = f_sigma(yaw_t)
        sat_bool = np.ones(len(num_sats))
        for i in range(len(num_sats)):
            if num_sats[i] <= 4:
                sat_bool[i] = 0
        yaw_results = (optimization.minimize(yaw_fit, 0,
                args = (temp_yaw, res_bearing, ins_unc, sat_bool)))
        offset = yaw_results.x[0]
        for i in range(len(temp_yaw)):
            temp_yaw[i] += offset

        print('Offset Yaw: ', offset)
        unc_threshold = (3 / 4) * max(ins_unc)
        index = np.where(ins_unc >= unc_threshold)

        f_cax = interp1d(ins_time, corrected_ax, axis = 0, fill_value = 'extrapolate')
        f_cay = interp1d(ins_time, corrected_ay, axis = 0, fill_value = 'extrapolate')
        res_ax = f_cax(yaw_t)
        res_ay = f_cay(yaw_t)

        res_ax = [np.mean()]

    if plot:
        plt.figure(figsize = (12,8))
        plt.plot(ins_time[:-1], new_bearing, label = 'INS Bearing', color = colors['ins'])
        plt.plot(yaw_t, temp_yaw, label = 'IMU Bearing', color = colors['imu'])
        plt.xlabel('Time [s]')
        plt.ylabel('Angles [Degrees]')
        plt.legend()
        plt.tight_layout()
        plt.savefig('/home/jacob/DNDO/IMU_GPS/' + name + '/imu_ins_bearings')
        plt.close()
        # plt.show()

        initial_avg = np.mean(temp_yaw[0:10])
        final_res_yaw = [i - initial_avg for i in temp_yaw]

        if lidar_active:
            plt.figure(figsize=(12,8))
            plt.plot(new_t, np.degrees(lidar_angles[:,2]), label = 'Lidar Yaw', color = 'blue')
            plt.plot(yaw_t, final_res_yaw, label = 'IMU Yaw', color = colors['imu'])
            plt.xlabel('Time [s]')
            plt.ylabel('Angle [Degrees]')
            plt.legend()
            plt.tight_layout()
            plt.savefig('/home/jacob/DNDO/IMU_GPS/' + name + '/lidar_imu_yaw')
            plt.close()
            # plt.show()

        plt.figure(figsize = (8,10))
        axes = plt.axes()
        plt.plot(vec_locations[:,0], vec_locations[:,1], colors['ins'], label = 'INS')
        for i in range(len(new_t)):
            if i == 190:
                axes.arrow(arrows[i,0], arrows[i,1], arrows[i,2], arrows[i,3], head_width=4, head_length=3, fc='r', ec='r')
            else:
                axes.arrow(arrows[i,0], arrows[i,1], arrows[i,2], arrows[i,3], head_width=4, head_length=3, fc='k', ec='k')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('/home/jacob/DNDO/IMU_GPS/' + name + '/imu_trajectory_over_ins')
        plt.close()
        # plt.show()
        break
    #########################################################################################################################################################################################################
    # Find change in sign of slopes for lidar and INS
    if lidar_active:
        lidar_dxy = np.zeros(len(new_t))
        for i in range(len(new_t) - 1):
            lidar_dxy[i] = np.sqrt((lidar_loc[i+1,0] - lidar_loc[i,0])**2 + (lidar_loc[i+1,1] - lidar_loc[i,1])**2)

        vec_dxy = np.zeros(len(new_t))
        for i in range(len(new_t) - 1):
            vec_dxy[i] = np.sqrt((resampled_loc[i+1,0] - resampled_loc[i,0])**2 + (resampled_loc[i+1,1] - resampled_loc[i,1])**2)

        multiplier = 2
        initial_guess = [0, 0, 0]
        sat_bool = np.ones(len(num_sats))
        counter = 0
        for i in range(len(num_sats)):
            if num_sats[i] <= 4 or vec_dxy[i] >= (multiplier * lidar_dxy[i]) + 4:
                sat_bool[i] = 0
                counter += 1

        j = 0
        important_locs = np.zeros((len(sat_bool)-counter, 2))
        for i in range(len(sat_bool)):
            if sat_bool[i] == 1:
                important_locs[j] = [new_lat[i], new_long[i]]
                j += 1

        sigma_g = np.ones(len(new_t))
        if unc:
            sigma_g = f_sigma(new_t)

        methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP']

        flag3 = time.time()
        results = optimization.minimize(traj_fit, initial_guess, method = methods[8],
                args = (lidar_loc, resampled_loc, sat_bool, sigma_g))
        flag4 = time.time()

        lidar_final_xy = np.zeros((len(new_t), 2))
        for i in range(len(new_t)):
            lidar_final_xy[i] = rotate([0,0], lidar_loc[i], np.radians(results.x[2]))

        for i in range(len(new_t)):
            lidar_final_xy[i] = [lidar_final_xy[i,0] + results.x[0], lidar_final_xy[i,1] + results.x[1]]

        print('--------------------------------------------')
        print(results.message)
        print('Angle Rotated: ', round(results.x[2], 2))
        print('Delta X: ', round(results.x[0], 2))
        print('Delta Y: ', round(results.x[1], 2))

    if plot:
        if lidar_active:
            plt.figure(figsize = (8,10))
            plt.plot(lidar_final_xy[:,0], lidar_final_xy[:,1], color = colors['lidar'], label = 'Lidar')
            plt.plot(vec_locations[:,0], vec_locations[:,1], color = colors['ins'], label = 'Vectornav')
            plt.xlabel('x (m)')
            plt.ylabel('y (m)')
            plt.legend()
            plt.tight_layout()
            plt.savefig('/home/jacob/DNDO/IMU_GPS/' + name + '/lidar_vectornav_traj')
            plt.close()
            # plt.show()

            for i in range(len(yaw_t)):
                arrows[i,0] = lidar_final_xy[N*i,0]
                arrows[i,1] = lidar_final_xy[N*i,1]

            plt.figure(figsize = (8,10))
            axes = plt.axes()
            plt.plot(lidar_final_xy[:,0], lidar_final_xy[:,1], color = colors['lidar'], label = 'Lidar')
            for i in range(len(new_t)):
                axes.arrow(arrows[i,0], arrows[i,1], arrows[i,2], arrows[i,3], head_width=3, head_length=2, fc='k', ec='k')
            plt.xlabel('x (m)')
            plt.ylabel('y (m)')
            plt.legend()
            plt.tight_layout()
            plt.savefig('/home/jacob/DNDO/IMU_GPS/' + name + '/imu_trajectory_over_lidar')
            plt.close()
            # plt.show()

        index = np.where(gps_fix == 0)
        plt.figure(figsize = (12,8))
        plt.plot(imu_time, ax, label = 'Acc. in X')
        plt.plot(imu_time, ay, label = 'Acc. in Y')
        plt.plot(imu_time, az, label = 'Acc. in Z')
        if type(index) == list:
            plt.axvline(imu_time[index[0][0]], color = 'red', linewidth = 2)
            plt.axvline(imu_time[index[0][-1]], color = 'red', linewidth = 2)
        plt.xlabel('Time [s]')
        plt.ylabel(r'Acceleration [$m/s^2$]')
        plt.legend()
        plt.tight_layout()
        plt.savefig('/home/jacob/DNDO/IMU_GPS/' + name + '/filtered_acc')
        plt.close()
        # plt.show()

    #########################################################################################################################################################################################################
    #  Reversal Rotation on Lidar to plot on google maps at GPS inital point
    #########################################################################################################################################################################################################
    if lidar_active:
        if not unc:
            fx = interp1d(ins_time, lat, axis = 0, fill_value = 'extrapolate')
            fy = interp1d(ins_time, long, axis = 0, fill_value = 'extrapolate')
        if unc:
            fx = interp1d(ins_time, lat, axis = 0, fill_value = 'extrapolate')
            fy = interp1d(ins_time, long, axis = 0, fill_value = 'extrapolate')

        lat = fx(new_t)
        long = fy(new_t)

        dx, dy = [], []
        for i in range(1,len(new_t)):
            dx.append(lidar_final_xy[i,0] - lidar_final_xy[i-1, 0])
            dy.append(lidar_final_xy[i,1] - lidar_final_xy[i-1, 1])

        lidar_gps = np.zeros((len(lidar_loc), 2))
        lidar_gps[0, 0] = lat[0] + results.x[1]  / (6371 * 1000) * (180 / np.pi)
        lidar_gps[0, 1] = long[0] + results.x[0]  / (6371 * 1000) * (180 / np.pi) / np.cos(lat[1] * np.pi / 180)

        for i in range(1,len(new_t)):
            lidar_gps[i,0] = lidar_gps[i-1, 0] + dy[i-1] / (6371 * 1000) * (180 / np.pi)
            lidar_gps[i,1] = lidar_gps[i-1, 1] + dx[i-1] / (6371 * 1000) * (180 / np.pi) / np.cos(lidar_gps[i,0] * np.pi / 180)

        #True start and end:  [Run1 INS], [Run2 INS], [Run1 GPS], [Run2 GPS]
        M = 1
        dummy = [[-3, -1.5, -10, 6], [2, -0, 3, -2], [-6.5, 1, -2, 2], [3, 1.3, 1, -2.2]]
        start_loc = [lat[0] + dummy[M][0] / (6371 * 1000) * (180 / np.pi),
            long[0] + dummy[M][1] / (6371 * 1000) * (180 / np.pi) / np.cos(lat[0] * np.pi / 180)]
        end_loc = [lat[-1] + dummy[M][2] / (6371 * 1000) * (180 / np.pi),
            long[-1] + dummy[M][3] / (6371 * 1000) * (180 / np.pi) / np.cos(lat[-1] * np.pi / 180)]

        # plot = True
        if plot:
            gmap = gmplot.GoogleMapPlotter(lidar_gps[0,0], lidar_gps[0,1], 18)
            gmap.scatter([start_loc[0]], [start_loc[1]], 'purple')
            gmap.scatter([end_loc[0]], [end_loc[1]], 'purple')
            gmap.plot(lat, long, colors['ins'], edge_width=7)
            gmap.plot(important_locs[:,0], important_locs[:,1], 'red', edge_width=7)
            gmap.scatter([lat[0]], [long[0]], 'green')
            gmap.scatter([lat[-1]], [long[-1]], 'red')
            gmap.plot(lidar_gps[:,0], lidar_gps[:,1], colors['lidar'], edge_width=7)
            gmap.scatter([lidar_gps[0,0]], [lidar_gps[0,1]], 'green')
            gmap.scatter([lidar_gps[-1,0]], [lidar_gps[-1,1]], 'red')
            # gmap.draw('/home/jacob/DNDO/IMU_GPS/' + name + '/both_traj_ins_resampled.html')
            gmap.draw('/home/jacob/DNDO/IMU_GPS/' + name + '/testing.html')
        # plot = False

        end = time.time()
        print('Time elapsed : ',end - start)
        print('Time for Traj: ', flag2 - flag1)
        print('Time for Fit : ', flag4 - flag3)

    if plot:
        plt.figure(figsize = (10, 8))
        plt.scatter(gps_time, pd_gps['NumSats'])
        plt.xlabel('Time [s]', fontsize = 14)
        plt.ylabel('Number of Satellites', fontsize = 14)
        plt.tight_layout()
        plt.savefig('/home/jacob/DNDO/IMU_GPS/' + name + '/number_sats')
        plt.close()

        plt.figure(figsize = (10, 8))
        plt.scatter(gps_time, pd_gps['GpsFix'])
        plt.xlabel('Time [s]', fontsize = 14)
        plt.ylabel('GPS Fix [Dimension]', fontsize = 14)
        plt.tight_layout()
        plt.savefig('/home/jacob/DNDO/IMU_GPS/' + name + '/gps_fix')
        plt.close()

        highest = max(pd_ins['Pos_Unc'])
        lowest = min(pd_ins['Pos_Unc'])
        sections = np.linspace(lowest, highest, 5)

        plt.figure(figsize = (10, 10))
        plt.scatter(vec_locations[:,0], vec_locations[:,1], c = pd_ins['Pos_Unc'], cmap = 'inferno')
        plt.xlabel('X - Loc. [m]', fontsize = 14)
        plt.ylabel('Y - Loc. [m]', fontsize = 14)
        cbar = plt.colorbar()
        cbar.ax.get_yaxis().set_ticks([])
        for j, lab in enumerate([str(sections[0]), str(sections[1]), str(sections[2]), str(sections[3]), str(sections[4])]):
            cbar.ax.text(-.5, (2 * j ) / 8.0, lab[:2], ha='center', va='center', fontsize = 14)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel('Uncertainty [m]', rotation=270, fontsize = 14)
        plt.tight_layout()
        plt.savefig('/home/jacob/DNDO/IMU_GPS/' + name + '/Position_Uncertainty')
        plt.close()

# res_yaw = f_yaw(yaw_t)
# arrows = np.zeros((len(new_t), 4))
# dist = 3
# for i in range(len(yaw_t)):
#     arrows[i,0] = resampled_loc[N*i,0]
#     arrows[i,1] = resampled_loc[N*i,1]
#     arrows[i,2] = dist * -np.sin(np.radians(res_yaw[i]))
#     arrows[i,3] = dist * np.cos(np.radians(res_yaw[i]))
#
# vel_x = [(resampled_loc[i+1,0] - resampled_loc[i,0]) / (yaw_t[i+1] -
#         yaw_t[i]) for i in range(len(res_yaw)-1)]
# vel_y = [(resampled_loc[i+1,1] - resampled_loc[i,1]) / (yaw_t[i+1] -
#         yaw_t[i]) for i in range(len(res_yaw)-1)]
#
# acc_x = [(vel_x[i+1] - vel_x[i]) / (yaw_t[i+1] - yaw_t[i]) for i in range(len(res_yaw)-2)]
# acc_y = [(vel_y[i+1] - vel_y[i]) / (yaw_t[i+1] - yaw_t[i]) for i in range(len(res_yaw)-2)]
#
# adj_acc_x = np.zeros((len(res_yaw)-3))
# adj_acc_y = np.zeros((len(res_yaw)-3))
# for i in range(len(res_yaw)-3):
#     adj_acc_x[i] = (acc_x[i] * np.sin(np.radians(res_yaw[i]))
#             + acc_y[i] * np.cos(np.radians(res_yaw[i])))
#     adj_acc_y[i] = (acc_y[i] * -np.sin(np.radians(res_yaw[i]))
#             + acc_x[i] * np.cos(np.radians(res_yaw[i])))
#
# f_time = interp1d(imu_time, az, axis = 0, fill_value = 'extrapolate')
# time = np.linspace(0,yaw_t[-1], len(yaw_t)-3)
#
# plt.figure(figsize = (10,6))
# # plt.plot(imu_time, ax, label = 'Acc. in X', color = 'red', alpha = 1)
# plt.plot(ins_time, corrected_ax, label = 'IMU Acc. X', color = 'red', alpha = 1)
# plt.plot(ins_time, corrected_ay, label = 'IMU Acc. Y', color = 'purple', alpha = 1)
# plt.plot(time, adj_acc_x, label = 'INS Acc. X', color = 'navy')
# plt.plot(time, adj_acc_y, label = 'INS Acc. Y', color = 'darkorange')
# # plt.scatter([yaw_t[150]], [0], s = 50, color = 'green', label = 'Comp')
# plt.plot(gps_time, pd_gps['NumSats'], linewidth = 3, color = 'green')
# plt.ylim(-10,10)
# plt.xlabel('Time [s]')
# plt.ylabel(r'Acceleration [$m/s^2$]')
# plt.legend()
# plt.tight_layout()
# plt.show()
