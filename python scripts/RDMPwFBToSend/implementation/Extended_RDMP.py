"""
@author Farzaneh.Taleb
9/6/20 5:59 PM
Implementation of Extended RDMP model
"""

import numpy as np



class RRDMP_complete:

    def __init__(self,p, q, d, m, prob1, prob2, l_block, n_block, alpha,flagBLA2ACC,
                                       flagACC2BLA,flagST2BLA,flagST,th,type_decision):
        self.alpha=alpha

        self.m = m
        self.N = 2 * m
        self.decay_rate_d = d
        self.p_array = p ** np.arange(1, m)
        self.destable_array = d ** np.arange(1, m)
        self.q_array = q ** (((m - 2) * np.arange(1, m + 1) + 1) / (m - 1))
        self.l_block = l_block
        self.prob1 = prob1
        self.prob2 = prob2
        self.n_block = n_block
        self.flagBLA2ACC = flagBLA2ACC
        self.flagACC2BLA = flagACC2BLA
        self.flagST2BLA = flagST2BLA
        self.flagST = flagST
        self.type_decision=type_decision

        self.stimulus, self.stim_prob = self.generate_input3(prob1, prob2, l_block, n_block)

        size = self.stim_prob.shape[1]

        self.long_avg,self.short_avg =self.initalize_global()
        self.deltaF = np.full(([size]), 0.25)
        self.deltaF_p = np.zeros(shape=[size])
        self.performance_acc = np.zeros(shape=[size])
        self.performance_rdmp = np.zeros(shape=[size])
        self.performance_st = np.zeros(shape=[size])
        self.d_val = np.zeros(shape=[size])
        self.deltaF_s = np.zeros(shape=[size])
        self.deltaF_w = np.zeros(shape=[size])
        self.deltaF_F = np.zeros(shape=[size])
        self.rpe = np.zeros(shape=[size])
        self.rew = np.zeros(shape=[size])
        self.rew_avg = np.zeros(shape=[size])
        # self.rpe = np.zeros(shape=[size])
        self.rpe_meta = np.zeros(shape=[size])
        self.rpe_meta_meanRemove = np.zeros(shape=[size])
        self.rpe_avg_meanRmv = np.zeros(shape=[size])
        self.rpe_avg_meta_meanRmv = np.zeros(shape=[size])
        self.rpe_avg = np.zeros(shape=[size])
        self.rpe_avg_meta = np.zeros(shape=[size])
        self.dif_meanRemov = np.zeros(shape=[size])
        self.dif = np.zeros(shape=[size])
        self.decision_prob = np.zeros(shape=[size])
        self.deltaFs_meanRmv = np.zeros(shape=[size])
        self.deltaFw_meanRmv = np.zeros(shape=[size])

        # self.v = np.full(([2,size]) , 0)
        self.v = np.full(([2, size]), 0.5)
        self.cnt = 0
        self.delta_f_B_strong = np.zeros([2, size])
        self.delta_f_B_weak = np.zeros([2, size])
        self.k_b_strong = np.zeros([2, size])
        self.k_b_weak = np.zeros([2, size])
        self.C_strong = np.full(([2, self.m]), 1 / self.N)
        self.C_weak = np.full(([2, self.m]), 1 / self.N)
        self.c_strong = np.full(([2, size, self.m]), 1 / self.N)
        self.c_weak = np.full(([2, size, self.m]), 1 / self.N)
        self.basealpha = 0.01
        self.estimated_prob = np.full(size, .5)
        self.p_extended_model = np.full([2, size], .5)
        self.p_st = np.full([2, size], .5)
        self.p_acc = np.full([2, size], .5)

        self.deltaF_t = np.full([size], 0.00)
        self.sum_deltaF_t = np.full([size], 0.00)
        self.c_p_acc = np.full([2, size], .5)

        self.p_r_acc = np.full([2, size], .5)
        self.p_r_st = np.full([2, size], .5)
        self.p_r_st3 = np.full(size, .5)
        self.destablization_count = np.zeros([size])
        self.pTp_meanRmv = np.full([size], float(th))
        self.pTp = np.full([size], float(th))
        self.pTp_meta = np.full([size], float(th))
        self.pTp_meta_meanRemove = np.full([size], float(th))

        self.exp_from_metaplastic_normalized = np.full([size], float(th))
        self.exp_from_plastic_normalized = np.full([size], float(th))
        self.th = th
        self.randoms =  np.random.rand(self.l_block)
        self.sum_deltaF = np.zeros([size])
        self.r = np.zeros([size])
        self.kb_s =  np.zeros([ size])
        self.kb_w =  np.zeros([ size])




    """
    delta_f :
    changes in the synaptic strength for synapses associated with one of the option
    when reward was assigned to that option

    k_b :
      the overall increase in the efficacy of metaplastic synapses associated with that option
      divided by the total fraction of those synapses in related meta-states 
    """


    def sigmoid(self, x, sigma):
        # print("X",x)
        p = 1 / (1 + np.exp(-x / sigma))
        return p

    def update_v(self, alpha, v, r):
        x = v + alpha * (r - v)
        return x

    def generate_input3(self, prob1, prob2, l_block, n_block):
        stimulus = np.zeros(shape=[2, l_block * n_block])
        probs = np.zeros(shape=[2, l_block * n_block])
        stimulus1 = []
        stimulus0 = []

        probs1 = []
        probs0 = []

        # prob=prob1
        for i in range(n_block):
            n_ones_1 = int(np.round(prob1 * l_block))
            n_ones_2 = int(np.round(prob2 * l_block))

            n_zeros_1 = l_block - n_ones_1
            n_zeros_2 = l_block - n_ones_2

            array1 = np.concatenate((np.zeros([n_zeros_1]), np.zeros([n_ones_1]) + 1))
            array2 = np.concatenate((np.zeros([n_zeros_2]), np.zeros([n_ones_2]) + 1))

            np.random.shuffle(array1)
            np.random.shuffle(array2)

            stimulus1 = np.concatenate((stimulus1, array1))
            stimulus0 = np.concatenate((stimulus0, array2))

            probs1 = np.concatenate((probs1, np.full(l_block, fill_value=prob1)))
            probs0 = np.concatenate((probs0, np.full(l_block, fill_value=prob2)))

            prob1, prob2 = prob2, prob1

        stimulus[0, :] = stimulus0
        stimulus[1, :] = stimulus1

        probs[0, :] = probs0
        probs[1, :] = probs1

        return np.asarray(stimulus), np.asarray(probs)

    def update_fractions_step_by_step(self, C_s, C_w):
        for i in range(self.m):
            if i == 0:
                C_s[i] = C_s[i] + self.q_array[i] * C_w[i]
                C_w[i] = C_w[i] - self.q_array[i] * C_w[i]
                C_s[i + 1] = C_s[i + 1] + self.p_array[i] * C_s[i]
                C_s[i] = C_s[i] - self.p_array[i] * C_s[i]
            elif i == self.m - 1:

                C_w[i - 1] = C_w[i - 1] + self.p_array[i - 1] * C_w[i]
                C_w[i] = C_w[i] - self.p_array[i - 1] * C_w[i]
                C_s[0] = C_s[0] + self.q_array[i] * C_w[i]
                C_w[i] = C_w[i] - self.q_array[i] * C_w[i]
            else:
                C_s[i + 1] = C_s[i + 1] + self.p_array[i] * C_s[i]
                C_s[i] = C_s[i] - self.p_array[i] * C_s[i]
                C_w[i - 1] = C_w[i - 1] + self.p_array[i - 1] * C_w[i]
                C_w[i] = C_w[i] - self.p_array[i - 1] * C_w[i]
                C_s[0] = C_s[0] + self.q_array[i] * C_w[i]
                C_w[i] = C_w[i] - self.q_array[i] * C_w[i]
        return C_s, C_w

    def destable_steps(self, C_s, C_w):
        for i in range(self.m):
            if i == 0:
                pass
            else:
                C_w[i - 1] = C_w[i - 1] + self.destable_array[i - 1] * C_w[i]
                C_w[i] = C_w[i] - self.destable_array[i - 1] * C_w[i]
                C_s[i - 1] = C_s[i - 1] + self.destable_array[i - 1] * C_s[i]
                C_s[i] = C_s[i] - self.destable_array[i - 1] * C_s[i]

        return C_s, C_w

    def delta_f_option(self, f):
        delta_f_b = np.dot(self.q_array, f)
        k_b = delta_f_b / np.sum(f)
        return delta_f_b, k_b

    def simulate_step_by_step(self, t, lambdaa_longterm, lambdaa_shortTerm, decision_type):
        sig = 0.4
        self.p_acc[1, t] = self.sigmoid((np.sum(self.c_strong[1, t]) - np.sum(self.c_strong[0, t])), sig)
        self.p_acc[0, t] = 1 - self.p_acc[1, t]

        self.p_st[1, t] = self.sigmoid(self.v[1, t] - self.v[0, t], sig)
        self.p_st[0, t] = 1 - self.p_st[1, t]
        ###################

        self.p_r_acc[0, t] = np.sum(self.c_strong[0, t])
        self.p_r_acc[1, t] = np.sum(self.c_strong[1, t])
        self.p_r_st[0, t] = self.v[0, t]
        self.p_r_st[1, t] = self.v[1, t]

        epsilon= 0.1
        prob_select = np.random.rand(1)
        if prob_select < epsilon:
            select=np.random.randint(0,2)
        else:
            if decision_type==1:
                if self.p_st[0,t]>self.p_st[1,t]:
                    select=0
                else:
                    select=1
            elif decision_type==2 or decision_type==3:
                if self.p_acc[0,t]>self.p_acc[1,t]:
                    select=0
                else:
                    select=1

        self.rew[t] = self.stimulus[select, t]
        ###############
        self.pTp[t] = self.p_r_st[select, t] * (1 - self.p_r_st[select, t])

        self.pTp_meta[t] = self.p_r_acc[select, t] * (1 - self.p_r_acc[select, t])

        #todo update_fraction_steps ro check konam.
        f_c_strong_pot, f_c_weak_pot = self.update_fractions_step_by_step(self.c_strong[select, t].copy(),
                                                                          self.c_weak[select,t].copy())
        f_c_weak_dep, f_c_strong_dep = self.update_fractions_step_by_step(self.c_weak[select, t].copy(),
                                                                          self.c_strong[select,t].copy())
        v_pot = self.update_v(self.alpha, self.v[select, t], 1)
        v_dep = self.update_v(self.alpha, self.v[select, t], 0)
        self.deltaF_s[t] = np.sum(f_c_strong_pot - self.c_strong[select,t])
        self.deltaF_w[t] = np.sum(f_c_strong_dep - self.c_strong[select,t])
        self.deltaF_F[t] = self.deltaF_s[t] - self.deltaF_w[t]
        self.kb_s[t] = self.deltaF_s[t]/np.sum(self.c_weak[select,t])
        self.kb_w[t] = -self.deltaF_w[t]/np.sum(self.c_strong[select,t])

        if t+1 < self.v.shape[1]:
            # t = t + 1
            if self.rew[t] == 1:
                self.c_strong[select,t+1] = (f_c_strong_pot)
                self.c_weak[select,t+1] = (f_c_weak_pot)
                self.v[select, t+1] = v_pot

            else:
                self.c_strong[select,t+1] = (f_c_strong_dep)
                self.c_weak[select,t+1] = (f_c_weak_dep)
                self.v[select, t+1] = v_dep

            self.v[1 - select, t+1] = self.v[1 - select, t ]
            self.c_strong[1-select,t+1] = self.c_strong[1-select,t]
            self.c_weak[1-select,t+1] = self.c_weak[1-select,t]

            #full RDMP
            if self.flagBLA2ACC and self.flagACC2BLA and self.flagST2BLA and self.flagST :
                if t > 1:
                    self.critical(t,select,lambdaa_longterm,lambdaa_shortTerm)
                    if self.deltaF[t] - self.rpe_avg_meanRmv[t] > 0:
                        self.destablization_count[t] = 1
                        for i in range(2):
                            self.c_strong[i,t+1], self.c_weak[i,t+1] = self.destable_steps(self.c_strong[i,t+1].copy(),self.c_weak[i,t+1].copy())







            #Random Destabilization
            elif not self.flagBLA2ACC and self.flagACC2BLA and self.flagST2BLA and self.flagST :
                if t > 1:
                    if np.random.rand(1)<0.5:
                        self.destablization_count[t] = 1
                        for i in range(2):
                            self.c_strong[i, t + 1], self.c_weak[i, t + 1] = self.destable_steps(
                                self.c_strong[i, t + 1].copy(), self.c_weak[i, t + 1].copy())

            #static threshold with eu
            elif  self.flagBLA2ACC and not self.flagACC2BLA and self.flagST2BLA and self.flagST :
                if t > 1:
                    self.critical(t,select,lambdaa_longterm,lambdaa_shortTerm)
                    if self.rpe_avg_meanRmv[t]>self.th :
                        self.destablization_count[t] = 1
                        for i in range(2):
                            self.c_strong[i,t+1], self.c_weak[i,t+1] = self.destable_steps(self.c_strong[i,t+1].copy(),self.c_weak[i,t+1].copy())

            #static threshold with uu mean removed
            elif self.flagBLA2ACC and self.flagACC2BLA and not self.flagST2BLA and self.flagST:
                if t>1:
                    self.critical(t, select, lambdaa_longterm, lambdaa_shortTerm)

                    if self.deltaF[t] - self.th > 0:
                        self.destablization_count[t] = 1
                        for i in range(2):
                            self.c_strong[i, t + 1], self.c_weak[i, t + 1] = self.destable_steps(
                                self.c_strong[i, t + 1].copy(), self.c_weak[i, t + 1].copy())

           #static threshold with uu not mean removed
            elif self.flagBLA2ACC and self.flagACC2BLA and  self.flagST2BLA and not self.flagST:

                if self.deltaF[t] - self.th > 0:
                    self.destablization_count[t] = 1
                    for i in range(2):
                        self.c_strong[i, t + 1], self.c_weak[i, t + 1] = self.destable_steps(
                            self.c_strong[i, t + 1].copy(), self.c_weak[i, t + 1].copy())

           #RDMP with compuatation of everuthing but not destabilization
            elif not self.flagBLA2ACC and not self.flagACC2BLA and not self.flagST2BLA and self.flagST:
                if t > 1:
                   self.critical(t,select,lambdaa_longterm,lambdaa_shortTerm)
            else:
                print("NOTHING")

    def run_trials(self):
        time=self.l_block*self.n_block
        for t in range(time):
            # self.simulate_steppp(t, 0.95, 0.45, self.type_decision)
            self.simulate_step_by_step(t, 0.85, 0.4, self.type_decision)

        return self


    def initalize_global(self):
        # time_max_long = 1000
        # tt = 1000
        # ws_longTerm = [1]
        # ws_longTrem_temp = np.geomspace(0.95, np.power(0.95, tt - 1), num=tt - 1)
        # ws_longTerm = np.concatenate((ws_longTerm, ws_longTrem_temp))
        #
        tt = 709
        ws_longTerm = [1]
        ws_longTrem_temp = np.geomspace(0.85, np.power(0.85, tt - 1), num=tt - 1)
        ws_longTerm = np.concatenate((ws_longTerm, ws_longTrem_temp))


        tt = 709
        ws_shortTerm = [1]
        ws_shortTerm_temp = np.geomspace(0.4, np.power(0.4, tt), num=tt)
        ws_shortTerm = np.concatenate((ws_shortTerm, ws_shortTerm_temp))

        return ws_longTerm,ws_shortTerm

    def critical(self,t,select,lambdaa_longterm,lambdaa_shortTerm):
        time_max_long = 709
        ws_longTerm = [1]
        tt = min(t, time_max_long)
        if tt < time_max_long:
            ws_longTrem_temp = np.geomspace(lambdaa_longterm, np.power(lambdaa_longterm, tt - 1),
                                            num=tt - 1)
            ws_longTerm = np.concatenate((ws_longTerm, ws_longTrem_temp))
        else:
            ws_longTerm = self.long_avg
        ws_longTerm = np.concatenate((ws_longTerm, np.zeros(shape=[t - tt])))
        ws_longTerm = np.flip(ws_longTerm)


        self.deltaFs_meanRmv[t] = self.deltaF_s[t] - np.average(self.deltaF_s[:t], weights=np.transpose(ws_longTerm))
        self.deltaFw_meanRmv[t] = self.deltaF_w[t] - np.average(self.deltaF_w[:t], weights=np.transpose(ws_longTerm))
        self.deltaF[t] = self.deltaFs_meanRmv[t] + self.deltaFw_meanRmv[t]


        self.rpe[t] = np.abs(self.rew[t] - self.p_r_st[select, t])
        self.rpe_meta[t] =np.abs(self.rew[t] - self.p_r_acc[select, t])
        self.rpe_avg_meanRmv[t] = self.rpe[t] - np.average(self.rpe[:t], weights=np.transpose(ws_longTerm))
        self.rpe_avg_meta_meanRmv[t] = self.rpe_meta[t] - np.average(self.rpe_meta[:t],weights=np.transpose(ws_longTerm))
        self.dif_meanRemov[t] = self.deltaF[t] - self.rpe_avg_meanRmv[t]
        self.dif[t] = self.deltaF_F[t] - self.rpe_avg[t]
        self.pTp_meanRmv[t] = self.pTp[t] - np.average(self.pTp[:t], weights=np.transpose(ws_longTerm))
        self.pTp_meta_meanRemove[t] = self.pTp_meta[t] - np.average(self.pTp_meta[:t], weights=np.transpose(ws_longTerm))

