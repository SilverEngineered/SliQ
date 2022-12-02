import numpy as np
import scipy.stats as sp
import matplotlib as mpl
import matplotlib.pyplot as mp

params = {
	'font.family': 'serif',
	'text.usetex': True,
	'pgf.rcfonts': False,
	'pgf.texsystem': 'xelatex',
	'pgf.preamble': r'\usepackage{fontspec,physics}',
}

mpl.rcParams.update(params)

'''
data = [56.80393843, 29.41724861, 26.36365404, 57.21194746, 30.42203212, 29.90584404, 52.61051743, 17.85698497, 70.62200427, 40.32504729, 24.88866326, 43.56527316, 99.33934927, 2.524367581, 37.8158601, -20.2400674, 27.72936074, 22.21948769, 58.6810296, 23.12222656, 54.53423347, 46.84021906, -6.264096209, -11.28961864, 17.03351365, 9.395716731, 16.17467603, 21.96207012, 64.40156334, 15.50477994, 51.03997028, 28.59905066, 61.87933527, 28.34214345]
data = np.histogram(data, bins=np.arange(int(min(data))-1, max(data)+1, 1))
bins = data[1]
data = data[0]
data = np.cumsum(data)

fig = mp.figure(figsize=(5, 2.5))

ax = fig.add_subplot(111)
ax.grid(color='lightgrey', linestyle=':')
ax.set_axisbelow(True)

ax.plot(bins[:-1], data, linewidth=1.5, color='red')

ax.set_xlim([bins[0], bins[-1]])
ax.set_ylim([data[0], data[-1]])

ax.set_yticks(np.linspace(data[0], data[-1], 5))
ax.set_yticklabels(['0', '25', '50', '75', '100'])

mp.setp(ax.get_xticklabels(), fontsize=14)
mp.setp(ax.get_yticklabels(), fontsize=14)

ax.set_xlabel('\% Imp. Over the Mean', fontsize=14)
ax.set_ylabel('Empirical CDF (\%)', fontsize=14)

mp.tight_layout(pad=0.2, h_pad=0.2)
mp.savefig('imp_over_mean.pdf')
mp.close()

data = [1.641641144, 1.064976377, 0.7423226804, 1.437049282, 1.004105808, 1.08873523, 1.65582574, 0.6399093071, 2.280679443, 1.291356657, 0.8814579735, 2.259816956, 0.9583045964, 0.1492253993, 1.037890136, -0.8357374765, 1.194450389, 0.6357547398, 1.979760503, 0.7526057304, 1.890333269, 1.680325241, -0.2093764203, -0.4808988267, 0.5402243974, 0.5693612172, 0.5845316515, 0.8519255477, 2.443683089, 0.9061765029, 1.367421814, 0.9676369148, 1.778131996, 0.8828546189]
data = [d*100 for d in data]
data = np.histogram(data, bins=np.arange(int(min(data))-1, max(data)+1, 1))
bins = data[1]
data = data[0]
data = np.cumsum(data)

fig = mp.figure(figsize=(5, 2.5))

ax = fig.add_subplot(111)
ax.grid(color='lightgrey', linestyle=':')
ax.set_axisbelow(True)

ax.plot(bins[:-1], data, linewidth=1.5, color='red')

ax.set_xlim([bins[0], bins[-1]])
ax.set_ylim([data[0], data[-1]])

ax.set_yticks(np.linspace(data[0], data[-1], 5))
ax.set_yticklabels(['0', '25', '50', '75', '100'])

mp.setp(ax.get_xticklabels(), fontsize=14)
mp.setp(ax.get_yticklabels(), fontsize=14)

ax.set_xlabel('Coefficient of Variance (\%)', fontsize=14)
ax.set_ylabel('Empirical CDF (\%)', fontsize=14)

mp.tight_layout(pad=0.2, h_pad=0.2)
mp.savefig('cov.pdf')
mp.close()
'''

#np.save('all', {'no_inter': no_inter, 'no_inter_no_cons': no_inter_no_cons, 'no_cons': no_cons, 'good': good})

jump = 0.01
dat = np.load('all.npy', allow_pickle=True).item()

data = dat['no_cons']
data = np.histogram(data, bins=np.arange(int(min(data)), max(data)+jump, jump))
bins = data[1]
data = data[0]
data = np.cumsum(data)

fig = mp.figure(figsize=(5.8, 2.3))

ax = fig.add_subplot(111)
ax.grid(color='lightgrey', linestyle=':')
ax.set_axisbelow(True)

ax.plot(bins[:-1], data, linewidth=1.5, color='mediumturquoise', label=r'\textsc{SliQ}' + ' w/o Projection Variance Mitigation', linestyle='-.')

dat3 = dat['no_inter']
dat3 = np.histogram(dat3, bins=bins)
dat3 = dat3[0]
dat3 = np.cumsum(dat3)
ax.plot(bins[:-1], dat3, linewidth=1.5, color='salmon', label=r'\textsc{SliQ}' + ' w/o Interweaving', linestyle=':')




dat2 = dat['good']
dat2 = np.histogram(dat2, bins=bins)
dat2 = dat2[0]
dat2 = np.cumsum(dat2)
#chartreuse
ax.plot(bins[:-1], dat2, linewidth=1.5, color='maroon', label=r'\textsc{SliQ}')

#ax.set_xlim([bins[0], bins[-1]])
ax.set_xlim([bins[0], 1.5])
ax.set_ylim([data[0], data[-1]])

ax.set_yticks(np.linspace(data[0], data[-1], 5))
ax.set_yticklabels(['0', '25', '50', '75', '100'])

mp.setp(ax.get_xticklabels(), fontsize=14)
mp.setp(ax.get_yticklabels(), fontsize=14)

ax.set_xlabel('Projection Variance', fontsize=14)
ax.set_ylabel('Empirical CDF (\%)', fontsize=14)

ax.legend(ncol=2, edgecolor='black', loc='lower right', bbox_to_anchor=(0., 1.02, 1., .102),
		  mode='expand', borderaxespad=0.1, fontsize=14, handletextpad=0.5)

mp.tight_layout(pad=0.2, h_pad=0.2)
mp.savefig('precision_loss.pdf')
mp.close()
exit()


dat = np.load('rankings.npy', allow_pickle=True).item()

pearsons = []
spearmans = []

for i in range(len(dat['all_distances'])):

	dists = []
	losses = []

	for dist, img1 in dat['all_distances'][i][:30]:
		for loss, img2 in dat['all_losses'][i]:
			if img1 == img2:
				dists.append(dist)
				losses.append(loss)
				break

	#print(dists, losses)

	mxd, mnd, mxl, mnl = max(dists), min(dists), max(losses), min(losses)

	dists = [mxl - ((mxd - d)/(mxd - mnd)) * (mxl - mnl) for d in dists]

	p = sp.pearsonr(dists, losses)
	s = sp.spearmanr(dists, losses)

	#print('Pearson Correlation:', p[0], ', p-value:', p[1])
	#print('Spearman Correlation:', s[0], ', p-value:', s[1])

	pearsons.append(p[0])
	spearmans.append(s[0])

	if s[0] > 0.3: #if s[0] > -0.2 and s[0] < 0.2:
		fig = mp.figure(figsize=(2.6, 2.6))

		ax = fig.add_subplot(111)
		ax.grid(color='lightgrey', linestyle=':')
		ax.set_axisbelow(True)

		ax.scatter(dists, losses, color='tomato', edgecolor='black')

		slope, intercept, r, p, se = sp.linregress(dists, losses)

		ax.plot([0, 5], [intercept, slope*5 + intercept], linewidth=1, color='black', zorder=0)

		ax.set_xlim([min(dists), max(dists)])
		ax.set_ylim([min(losses), max(losses)])

		mp.setp(ax.get_xticklabels(), fontsize=14)
		mp.setp(ax.get_yticklabels(), fontsize=14)

		ax.set_xlabel('Ground Truth Distance', fontsize=14)
		ax.set_ylabel(r'\textsc{SliQ}' + '\'s Calculated Loss', fontsize=14)
		ax.set_title(f'Correlation = {round(s[0],2)}', fontsize=14)
		mp.tight_layout(pad=0.2, h_pad=0.2)
		mp.savefig('distance_vs_loss_' + str(round(s[0]*100)) + '.pdf')
		mp.close()

print('Num Samples', i)
print()
print('P25', np.percentile(pearsons, 25))
print('P50', np.percentile(pearsons, 50))
print('P75', np.percentile(pearsons, 100))
print()
print('S25', np.percentile(spearmans, 25))
print('S50', np.percentile(spearmans, 50))
print('S75', np.percentile(spearmans, 75))
print('S100', np.percentile(spearmans, 100))
'''

data = [[62.8, 81.89, 52, 71.54], [91.7, 88.35, 99.6, 97.28], [85.4, 82.30, 94, 91.8]]

titles = ['Aids', 'MNIST', 'Fashion-MNIST']
methods = ['Baseline', 'PQK', 'Quilt', r'\textsc{SliQ}']
colors = ['white', 'pink', 'tomato', 'maroon']

fig = mp.figure(figsize=(6, 2.8))

for i in range(len(data)):
	ax = fig.add_subplot(int('1' + str(len(data)) + str(i+1)))
	ax.yaxis.grid(color='lightgrey', linestyle=':')
	ax.set_axisbelow(True)

	for j in range(len(methods)):
		ax.bar([j], [data[i][j]], width=0.8, linewidth=0.5, edgecolor='black', color=colors[j])

	ax.set_xlim([-0.6, len(methods)-0.4])
	ax.set_ylim([0, 100])

	ax.set_xticks(range(len(methods)))
	ax.set_xticklabels(methods, rotation=90)

	ax.set_yticks([0, 20, 40, 60, 80, 100])

	mp.setp(ax.get_xticklabels(), fontsize=14)
	mp.setp(ax.get_yticklabels(), fontsize=14)

	ax.set_ylabel('Accuracy (\%)', fontsize=14)

	ax.set_title(titles[i], fontsize=14)

mp.tight_layout(pad=0.2, h_pad=0.2)
mp.savefig('main.pdf')
mp.close()
exit()

data=[
0.0806961719275446,
0.04942857754713061,
0.0006760190680387905,
-0.03575560756665584,
-0.014944336898456857,
-0.0006594641570900594,
-0.05242911736062465,
0.01609606181065366,
-0.028010750527413354,
-0.009087585787287673,
-0.05412633775002006,
-0.07504192106311533,
-0.037592477529215663,
-0.055121916060890376,
-0.04968619121877583,
-0.051490626373175905,
-0.08146171410498092,
-0.07952308686511043,
-0.08365143783777887,
-0.10541597307474915,
-0.06128269151237431,
-0.0953518991387726,
-0.07390691564260814,
-0.0872052499886366,
-0.09446720172437692,
-0.09786103789500258
]

mn, mx = min(data), max(data)

data = [1 - ((mx - d)/(mx - mn)) for d in data]

data2 = [
0.0004174994933658775,
-0.006048629684258391,
-0.004148571666479075,
-0.006378835274253496,
-0.00827551146013903,
-0.005509494533672686,
-0.006562700075252266,
-0.01663994551200202,
-0.014605255004060727,
-0.009871008184139423,
-0.010446806842791586,
-0.011947639581587824,
-0.013644333055840202,
-0.011056278992189746,
-0.01648855094165339,
-0.011998242074464116,
-0.015759035519009933,
-0.011706336973802873,
-0.012144604849121738,
-0.011799754654467967,
-0.011856097952971444,
-0.013540099543174722,
-0.0154823848816325,
-0.013058789897005731,
-0.015058911829149788,
-0.015183475913829381]
mn, mx = min(data2), max(data2)
data2 = [1 - ((mx - d)/(mx - mn)) for d in data2]
fig = mp.figure(figsize=(5.8, 2.0))

ax = fig.add_subplot(111)
ax.grid(color='lightgrey', linestyle=':')
ax.set_axisbelow(True)

ax.plot(range(0, len(data)*20, 20), data2, linewidth=1.5, color='orange', label='Baseline', linestyle='-.')

ax.plot(range(0, len(data)*20, 20), data, linewidth=1.5, color='maroon', label=r'\textsc{SliQ}')

ax.set_xlim([0, (len(data)-1)*20])
ax.set_ylim([0, 1])

ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_xticks(range(0, len(data)*20, 50))

mp.setp(ax.get_xticklabels(), fontsize=14)
mp.setp(ax.get_yticklabels(), fontsize=14)

ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('Training Loss', fontsize=14)

ax.legend(ncol=2, edgecolor='black', loc='upper right', #bbox_to_anchor=(0., 1.02, 1., .102),
		  borderaxespad=0.2, fontsize=14, handletextpad=0.5)

mp.tight_layout(pad=0.2, h_pad=0.2)
mp.savefig('training_loss.pdf')
mp.close()
'''

