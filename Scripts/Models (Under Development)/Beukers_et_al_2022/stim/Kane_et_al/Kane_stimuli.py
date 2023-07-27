r"""
Stimuli from `Kane et al., 2007 <https://psycnet.apa.org/record/2007-06096-010?doi=1>`_
Constructed from unique stimuli:  B, F, H, K, M, Q, R, X

From Kane et al:
Control targets are designated with 1\1\0 and there are 6 of them
Experimental targets are designated with 1\2\0 and there are 2 of them
Control foils are designated with 2\1\0 and there are 34 of them
All experimental foils are designated with 2\2\0 and there are 6 of them

*** 48 trials per block
*** Each letter should appear as a target once (i.e., 8 targets; 6 control, 2 experimental)
*** The 2 experimental target letters should not be used as experimental target letters in lists b, c, or d (the 2 experimental target letters in this list are B and F)
*** Each letter should appear as an experimental foil once if possible (the 6 in this list are B, F, H, K,M, R, and X)
*** Each letter should appear in the list 6 times
*** There should not be any 3-back lures

"""

Nback2_stims_a = ['Q', 'F', 'B', 'R', 'X', 'X', 'X', 'M', 'M', 'K', 'B', 'B', 'M', 'Q', 'M', 'X',
               'H', 'B', 'H', 'X', 'K', 'Q', 'F', 'F', 'F', 'K', 'K', 'M', 'R', 'H', 'H', 'M',
               'B', 'R', 'B', 'F', 'Q', 'H', 'Q', 'R', 'F', 'R', 'H', 'K', 'X', 'K', 'R', 'Q']
Nback2_stims_b = ['R', 'Q', 'H', 'K', 'F', 'F', 'R', 'B', 'B', 'B', 'F', 'M', 'K', 'H', 'X', 'B',
               'X', 'H', 'Q', 'H', 'F', 'K', 'Q', 'Q', 'Q', 'K', 'M', 'K', 'R', 'X', 'R', 'B',
               'M', 'H', 'M', 'R', 'R', 'F', 'X', 'F', 'B', 'H', 'K', 'M', 'M', 'Q', 'X', 'X']
Nback2_stims_c = ['F', 'X', 'H', 'M', 'F', 'X', 'X', 'M', 'H', 'F', 'Q', 'R', 'Q', 'B', 'B', 'M',
               'X', 'M', 'F', 'H', 'F', 'K', 'M', 'H', 'H', 'H', 'B', 'Q', 'B', 'K', 'K', 'K',
               'R', 'B', 'R', 'X', 'Q', 'X', 'M', 'K', 'R', 'R', 'F', 'Q', 'Q', 'K', 'R', 'B']
Nback2_stims_d = ['K', 'F', 'X', 'B', 'R', 'H', 'Q', 'Q', 'K', 'F', 'K', 'M', 'R', 'R', 'R', 'X',
               'B', 'X', 'Q', 'R', 'Q', 'K', 'K', 'X', 'R', 'B', 'H', 'B', 'F', 'F', 'H', 'B',
               'H', 'M', 'M', 'M', 'Q', 'F', 'X', 'F', 'B', 'H', 'H', 'M', 'K', 'Q', 'X', 'M']
Nback2_stims_e = ['F', 'M', 'Q', 'H', 'B', 'R', 'B', 'F', 'M', 'F', 'X', 'R', 'R', 'F', 'X', 'B',
               'X', 'Q', 'K', 'K', 'H', 'Q', 'B', 'Q', 'K', 'X', 'K', 'Q', 'Q', 'R', 'M', 'R',
               'H', 'H', 'H', 'B', 'B', 'K', 'F', 'X', 'M', 'M', 'M', 'R', 'X', 'F', 'H', 'K']
Nback2_stims_f = ['Q', 'H', 'K', 'M', 'Q', 'Q', 'F', 'K', 'F', 'X', 'X', 'M', 'R', 'M', 'H', 'B',
               'H', 'M', 'K', 'K', 'K', 'B', 'R', 'B', 'M', 'X', 'Q', 'X', 'R', 'B', 'H', 'H',
               'X', 'B', 'F', 'Q', 'H', 'Q', 'F', 'F', 'B', 'X', 'K', 'R', 'R', 'F', 'R', 'M']
Nback2_stims_g = ['R', 'B', 'Q', 'F', 'X', 'X', 'X', 'K', 'B', 'K', 'X', 'H', 'R', 'H', 'F', 'M',
               'F', 'H', 'B', 'B', 'M', 'H', 'M', 'Q', 'F', 'F', 'K', 'M', 'Q', 'Q', 'Q', 'R',
               'X', 'K', 'K', 'R', 'H', 'R', 'M', 'M', 'X', 'B', 'Q', 'B', 'H', 'K', 'F', 'R']
Nback2_stims_h = ['R', 'X', 'K', 'Q', 'R', 'M', 'M', 'H', 'B', 'H', 'F', 'F', 'F', 'X', 'K', 'Q',
               'R', 'R', 'K', 'B', 'K', 'R', 'H', 'R', 'Q', 'F', 'Q', 'K', 'H', 'H', 'F', 'X',
               'X', 'M', 'B', 'M', 'H', 'Q', 'B', 'B', 'B', 'M', 'F', 'X', 'K', 'X', 'Q', 'M']

Nback2_conds_a = ['2\1\0', '2\1\0', '2\1\0', '2\1\0', '2\1\0', '2\2\0', '1\2\0', '2\1\0',
                 '2\2\0', '2\1\0', '2\1\0', '2\2\0', '2\1\0', '2\1\0', '1\1\0', '2\1\0',
                 '2\1\0', '2\1\0', '1\1\0', '2\1\0', '2\1\0', '2\1\0', '2\1\0', '2\2\0',
                 '1\2\0', '2\1\0', '2\2\0', '2\1\0', '2\1\0', '2\1\0', '2\2\0', '2\1\0',
                 '2\1\0', '2\1\0', '1\1\0', '2\1\0', '2\1\0', '2\1\0', '1\1\0', '2\1\0',
                 '2\1\0', '1\1\0', '2\1\0', '2\1\0', '2\1\0', '1\1\0', '2\1\0', '2\1\0']
