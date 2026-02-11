// ── Client-Side Learner Model ────────────────────────────────────────────────
// Tracks per-concept accuracy across Bloom's levels, classifies mastery,
// and provides stats for the mastery dashboard and review prioritization.
// Factory function — call with a course slug to get a model instance.
//
// Usage: var lm = __lxpCreateLearnerModel('my-course');
//        lm.recordAnswer(['bond', 'yield'], 'apply', true);
//        var stats = lm.getStats();

function __lxpCreateLearnerModel(courseSlug) {
    var LEARNER_KEY = 'lxp_' + courseSlug + '_learner';

    var EMPTY_MODEL = {
        concepts: {},
        sessions: [],
        overall: {
            total_attempts: 0,
            total_correct: 0,
            accuracy: 0,
            concepts_mastered: 0,
            concepts_struggling: 0
        }
    };

    function load() {
        try {
            var raw = localStorage.getItem(LEARNER_KEY);
            if (!raw) return JSON.parse(JSON.stringify(EMPTY_MODEL));
            return JSON.parse(raw);
        } catch (e) {
            return JSON.parse(JSON.stringify(EMPTY_MODEL));
        }
    }

    function save(model) {
        try { localStorage.setItem(LEARNER_KEY, JSON.stringify(model)); } catch (e) {}
    }

    function classifyMastery(concept) {
        if (!concept || concept.attempts === 0) return 'new';
        if (concept.accuracy >= 0.85 && concept.attempts >= 3) return 'mastered';
        if (concept.accuracy < 0.50 && concept.attempts >= 2) return 'struggling';
        return 'progressing';
    }

    function recordAnswer(concepts, bloomLevel, isCorrect, confidence, score) {
        if (!concepts || concepts.length === 0) return;
        var model = load();
        concepts.forEach(function (name) {
            if (!model.concepts[name]) {
                model.concepts[name] = {
                    attempts: 0, correct: 0, accuracy: 0,
                    bloom_levels: {}, last_seen: null, mastery: 'new',
                    hypercorrections: 0, total_score: 0, avg_score: 0
                };
            }
            var c = model.concepts[name];
            c.attempts++;
            if (isCorrect) c.correct++;
            c.accuracy = c.attempts > 0 ? c.correct / c.attempts : 0;
            // Track hypercorrections: high confidence but wrong
            if (typeof confidence === 'number' && confidence >= 0.7 && !isCorrect) {
                c.hypercorrections = (c.hypercorrections || 0) + 1;
            }
            // Track continuous scores (0.0–1.0)
            if (typeof score === 'number') {
                c.total_score = (c.total_score || 0) + score;
                c.avg_score = c.total_score / c.attempts;
            }
            if (bloomLevel) {
                if (!c.bloom_levels[bloomLevel]) {
                    c.bloom_levels[bloomLevel] = { attempts: 0, correct: 0 };
                }
                c.bloom_levels[bloomLevel].attempts++;
                if (isCorrect) c.bloom_levels[bloomLevel].correct++;
            }
            c.last_seen = new Date().toISOString();
            c.mastery = classifyMastery(c);
        });
        // Update overall stats
        model.overall.total_attempts++;
        if (isCorrect) model.overall.total_correct++;
        model.overall.accuracy = model.overall.total_attempts > 0
            ? model.overall.total_correct / model.overall.total_attempts : 0;
        var mastered = 0, struggling = 0;
        Object.keys(model.concepts).forEach(function (k) {
            var m = model.concepts[k].mastery;
            if (m === 'mastered') mastered++;
            if (m === 'struggling') struggling++;
        });
        model.overall.concepts_mastered = mastered;
        model.overall.concepts_struggling = struggling;
        save(model);
    }

    function getStats() {
        return load().overall;
    }

    function getConceptMastery() {
        return load().concepts;
    }

    function recordSession(durationSec, elementsCompleted, accuracy) {
        var model = load();
        model.sessions.push({
            date: new Date().toISOString().slice(0, 10),
            duration_sec: durationSec,
            elements_completed: elementsCompleted,
            accuracy: accuracy
        });
        save(model);
    }

    return {
        load: load,
        save: save,
        classifyMastery: classifyMastery,
        recordAnswer: recordAnswer,
        getStats: getStats,
        getConceptMastery: getConceptMastery,
        recordSession: recordSession
    };
}
