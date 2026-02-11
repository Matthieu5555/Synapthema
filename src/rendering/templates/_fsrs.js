// Shared FSRS-5 scheduling engine
// Minimal inline port of the Free Spaced Repetition Scheduler (FSRS v5).
// Reference: https://github.com/open-spaced-repetition/fsrs4anki
var FSRS = (function() {
    // Default FSRS-5 parameters
    var w = [0.4072, 1.1829, 3.1262, 15.4722, 7.2102, 0.5316, 1.0651, 0.0589,
             1.5330, 0.1544, 1.0339, 1.9926, 0.1100, 0.2956, 2.2688, 0.2315,
             3.0578, 0.4657, 0.5753];
    var DECAY = -0.5;
    var FACTOR = Math.pow(0.9, 1 / DECAY) - 1;

    function clamp(val, lo, hi) { return Math.min(hi, Math.max(lo, val)); }

    function initDifficulty(grade) {
        return clamp(w[4] - Math.exp(w[5] * (grade - 1)) + 1, 1, 10);
    }

    function initStability(grade) {
        return Math.max(w[grade - 1], 0.1);
    }

    function nextDifficulty(d, grade) {
        var newD = d - w[6] * (grade - 3);
        // Mean reversion
        return clamp(w[7] * initDifficulty(3) + (1 - w[7]) * newD, 1, 10);
    }

    function retrievability(elapsedDays, stability) {
        if (stability <= 0) return 0;
        return Math.pow(1 + FACTOR * elapsedDays / stability, DECAY);
    }

    function nextRecallStability(d, s, r, grade) {
        var hardPenalty = (grade === 2) ? w[15] : 1;
        var easyBonus = (grade === 4) ? w[16] : 1;
        return s * (1 + Math.exp(w[8]) * (11 - d) * Math.pow(s, -w[9]) *
               (Math.exp((1 - r) * w[10]) - 1) * hardPenalty * easyBonus);
    }

    function nextForgetStability(d, s, r) {
        return w[11] * Math.pow(d, -w[12]) * (Math.pow(s + 1, w[13]) - 1) *
               Math.exp((1 - r) * w[14]);
    }

    function nextInterval(stability) {
        return Math.max(1, Math.round(stability * FACTOR / (Math.pow(0.9, 1 / DECAY) - 1)));
    }

    function scheduleCard(card, grade) {
        var now = Date.now();
        var elapsedDays = card.lastReview ? (now - card.lastReview) / 86400000 : 0;

        if (card.reps === 0) {
            // New card — first review
            card.difficulty = initDifficulty(grade);
            card.stability = initStability(grade);
        } else {
            var r = retrievability(elapsedDays, card.stability);
            card.difficulty = nextDifficulty(card.difficulty, grade);
            if (grade === 1) {
                // Forgot
                card.stability = Math.max(nextForgetStability(card.difficulty, card.stability, r), 0.1);
                card.lapses++;
            } else {
                card.stability = nextRecallStability(card.difficulty, card.stability, r, grade);
            }
        }

        card.reps++;
        card.lastReview = now;
        var interval = nextInterval(card.stability);
        card.nextReview = now + interval * 86400000;
        return card;
    }

    return { scheduleCard: scheduleCard, retrievability: retrievability };
})();