# UFC Predictor - Feature Spec (v1)

## Purpose
Defines model input features for predicting UFC fight winners.
All features must be computable using information strictly before the fight date (no leakage).

---

## Prediction Unit
One row = one fight(fight_id)
Two competitors: Red corner and Blue corner.

**Label**
- target: `red_win` (1 if red wins, 0 if blue wins)
- exclude: NC / Draw (handled in filtering rules)

--- Global Rules
### As-of rule(no leakage)
For fight F as `fight_date`, only use fights with `prior_fight_date < fight_date`

### Windows
- L5 = last 5 prior fights (per fighter)
- Career = all prior fights (per fighter)

### Eligibility
Keep a fight for training/eval only if: 
- `red_prior_fights >= 5` AND `blue_prior_fights >= 5`

### Units
- duration: seconds
- rates: per minute or per 15 minutes
- time since last fight: days

### Missing data rules
- If required raw stats are missing for any of the last 5 fights: mark feature as null and drop row (v1)

---

## Required Source Tables
### fights
- fight_id
- event_date
- weight_class
- rounds_scheduled
- red_fighter_id
- blue_fighter_id
- winner (red/blue)
- fight_duration_sec

### fight_stat (per fighter per fight)
- fight_id
- fighter_id
- sig_str_landed
- sig_str_attempted
- total_str_landed
- total_str_attempted
- td_landed
- td_attempted
- control_sec

### fighters
- fighter_id
- height_in
- reach_in
- dob
- stance

--- 

## Feature Definitions


### Helper columns (not model inputs)

| Feature | Scope | Definition |
| --- | --- | --- |
| red_prior_fights | fight-level | number of red fighter fights prior to fight_date |
| blue_prior_fights | fight-level | number of blue fighter fights prior to fight_date |
| red_age_years | fight_level | (fight_date - red_dob) / 365.25 |
| blue_age_years | fight_level | (fight_date - blue_dob) / 365.25 |

---

## FORM — Last 5 prior fights (L5)
| Feature | Scope | Window | Formula | Notes |
|---|---|---|---|---|
| red_win_rate_l5 | red | L5 | wins_last/5 | wins counted from fight outcomes |
| blue_win_rate_l5 | blue | L5 | wins_last/5 | wins counted from fight outcomes |
| red_avg_fight_duration | red | L5 | mean(fight_duration_l5) | use actual fight duration |
| blue_avg_fight_duration | blue | L5 | mean(fight_duration_l5) | use actual fight duration |
| red_days_since_last_fight | red | career | (fight_date - most_recent_prior_fight_date).days | must have >=1 prior fight |
| blue_days_since_last_fight | blue | career | (fight_date - most_recent_prior_fight_date).days | must have >=1 prior fight |

## STRIKING — Last 5 prior fights (L5)
Let: 
- total_minutes_last5 = sum(duration_sec_last5) / 60

| Feature | Scope | Window | Formula |
|---|---|---|---|
| red_sig_str_landed_per_min_l5 | red | L5 | sum(sig_landed_last5) / max(total_minutes_last5, 1e-6) | |blue_sig_str_landed_per_min_l5 | blue | L5 | sum(sig_landed_last5) / max(total_minutes_last5, 1e-6) |
| red_sig_str_accuracy_l5 | red | L5 | sum(sig_landed_last5) / max(sum(sig_attempted_last5), 1) |
| blue_sig_str_accuracy_l5 | blue | L5 | sum(sig_landed_last5) / max(sum(sig_attempted_last5), 1) |


Optional (if available):
| Feature | Scope | Window | Formula |
|---|---|---|---|
| red_total_str_landed_per_min_l5 | red | L5 | sum(total_str_landed_last5) / max(total_minutes_last5, 1e-6) |
| blue_total_str_landed_per_min_l5 | blue | L5 | sum(total_str_landed_last5) / max(total_minutes_last5, 1e-6) |

---

## GRAPPLING - Last 5 prior fights (L5)
| Feature | Scope | Window | Formula |
|---|---|---|---|
| red_control_sec_per_15_l5 | red | l5 | sum(control_sec_l5) / max(total_minutes_last5, 1e-6) * 15 * 60 | 
| blue_control_sec_per_15_l5 | blue | l5 | sum(control_sec_l5) / max(total_minutes_last5, 1e-6) * 15 * 60 | 
| red_td_landed_per_15_l5 | red | L5 |  sum(td_landed_l5) / max(total_minutes_last5, 1e-6) * 15 * 60 |
| blue_td_landed_per_15_l5 | blue | L5 |  sum(td_landed_l5) / max(total_minutes_last5, 1e-6) * 15 * 60 |
| red_td_accuracy_l5 | red | L5 | sum(td_landed_last5) / max(sum(td_attempted_last5), 1) |
| blue_td_accuracy_l5 | blue | L5 | sum(td_landed_last5) / max(sum(td_attempted_last5), 1) |

TD defense requires opponent attempts vs fighter:
| Feature | Scope | Window | Formula | Dependency |
|---|---|---|---|---|
| red_td_defense_l5 | red | L5 | 1 - (opp_td_landed_on_red_last5 / max(opp_td_attempted_on_red_last5,1)) | requires opponent-by-fight join |
| blue_td_defense_l5 | blue | L5 | 1 - (opp_td_landed_on_blue_last5 / max(opp_td_attempted_on_blue_last5,1)) |  |

---

## Experience / Durability - Career (pre-fight)
| Feature | Scope | Window | Formula |
|---|---|---|---|
| red_total_prior_fights | red | career | count(prior_fights) |
| blue_total_prior_fights | blue | career | count(prior_fights) |
| red_finish_rate | red | career | count(KO_wins + sub_wins) / max(total_wins, 1) | 
| blue_finish_rate | blue | career | count(KO_wins + sub_wins) / max(total_wins, 1) | 
| red_times_finished | red | career | ko_losses + sub_losses |
| blue_times_finished | blue | career | ko_losses + sub_losses |

---

## MATCH UP DELTAS (Red - Blue)
Deltas are computed for:
- reach, height, age
- all l5 metrics above

| Feature | Formula |
|---|---|
| reach_diff | red_reach - blue_reach |
| height_diff | red_height - blue_height |
| age_diff | red_age_years - blue_age_years |
| sig_str_landed_per_min_diff_l5 | red_sig_str_landed_per_min_l5 - blue_sig_str_landed_per_min_l5 |
| sig_str_accuracy_diff_l5 | red_sig_str_accuracy_l5 - blue_sig_str_accuracy_l5 |
| control_sec_per_15_diff_l5 | red_control_sec_per_15_l5 - blue_control_sec_per_15_l5 |
| td_landed_per_15_diff_l5 | red_td_landed_per_15_l5 - blue_td_landed_per_15_l5 |
| td_accuracy_diff_l5 | red_td_accuracy_l5 - blue_td_accuracy_l5 |
| td_defense_diff_l5 | red_td_defense_l5 - blue_td_defense_l5 |

---

## Filters / Exclusions (v1)
Drop fights where:
- outcome is Draw or No Contest
- either fighter has <5 prior fights
- missing required stats for L5 window

---

## Versioning
- Spec version: v1
- Any change requires:
  - increment spec version
  - retrain model
  - record change in CHANGELOG.md