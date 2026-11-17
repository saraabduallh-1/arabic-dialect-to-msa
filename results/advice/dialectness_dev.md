# Corpus dialectness (tweets vs fusha), evaluated on dev

Tweet vocabulary: 938,584 forms (count >= 20).

| tokens | dialect n | other n | AUC |
|---|---|---|---|
| all dev tokens | 645 | 781 | 0.44 |
| forms train shows only unmarked | 17 | 292 | 0.454 |

| candidate list (tweet count >= 200, score > t, not a train dialect form) | size | unseen dev dialect words covered (of 293) |
|---|---|---|
| score > 2 | 8,844 | 39 |
| score > 4 | 2,803 | 24 |
| score > 8 | 698 | 9 |

Top candidates: الله، ولا، الا، لو، لكن، له، انت، هل، صوت، جدا، اليوم، يعني، السبب، الناس، حتي، نتيجه، كيف، النظر، الحق، نحن، نصف، قال، كنت، صغير، اخر، دفع، صور، لان، السلام، طبيعي، لهم، لن، والله، ممكن، هم، الرقم، وانا، تعلم، اقامه، وان، يقول، ده، حرق، السعوديه، الشعب، الان، ابدا، الف، ارض، افضل، وكل، بكل، لنا، الثالث، الحب، ليس، ذهب، مساعده، ابو، انه
