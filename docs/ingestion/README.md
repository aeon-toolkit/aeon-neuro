# EEG conversion and trigger units

`convert_eeg_to_bids` retains dimensionless native STIM channels, such as BioSemi
Status, using `arbitrary` in the generated BIDS channel table. This spelling
preserves the native MNE unit code when reading the output with MNE-BIDS.
Samples, trigger codes and voltage-channel units remain unchanged. No new
dependency or trigger extraction policy is introduced.
