"""scripts/plot_metrics.py's log parsing must handle both log eras it will ever see.

engine.py used to log the full DSC/IoU/accuracy/sensitivity/specificity set only
every config.val_interval epochs, and never logged per-image test DSC. It now
logs the former every epoch and does log the latter (see engine.py's PATCH
comments) -- but old run logs on disk still have the sparse/absent form, and
scripts/plot_metrics.py must parse both without a version flag. This pins that
against small synthetic log excerpts rather than the real ~2,300-line log, and
needs no GPU: it exercises only regex/text parsing, not the model.
"""

import os

from scripts.plot_metrics import (
    derive_train_loss_per_epoch,
    find_latest_run,
    parse_lr_per_epoch,
    parse_per_image_test_dice,
    parse_test_summary,
    parse_train_epoch_summary,
    parse_val_blocks,
)

# A real val block's confusion matrix spans two physical lines (numpy's ndarray
# repr embeds a raw newline), and the logging formatter timestamps only the
# first line -- so log text is genuinely multi-line here, not an artifact of
# this fixture.
SPARSE_VAL_LINE = '2026-01-01 00:00:01 - val epoch: 1, loss: 0.4733\n'
DENSE_VAL_LINE = (
    '2026-01-01 00:00:02 - val epoch: 2, loss: 0.2232, miou: 0.8235, '
    'f1_or_dsc: 0.9032, accuracy: 0.9655,                 specificity: 0.9764, '
    'sensitivity: 0.9145, confusion_matrix: [[7910042  191160]\n'
    ' [ 147767 1581431]]\n'
)
TEST_SUMMARY_LINE = (
    '2026-01-01 00:00:03 - test of best model, loss: 0.2453,miou: 0.8225, '
    'f1_or_dsc: 0.9026, accuracy: 0.9643,                 specificity: 0.9808, '
    'sensitivity: 0.8917, confusion_matrix: [[31403838   614506]\n'
    ' [  790663  6512593]]\n'
)


def test_val_blocks_sparse_and_dense():
    blocks = parse_val_blocks(SPARSE_VAL_LINE + DENSE_VAL_LINE)
    assert [b['epoch'] for b in blocks] == [1, 2]

    sparse = blocks[0]
    assert sparse['loss'] == 0.4733
    assert sparse['dsc'] is None and sparse['miou'] is None

    dense = blocks[1]
    assert dense['loss'] == 0.2232
    assert dense['dsc'] == 0.9032
    assert dense['miou'] == 0.8235
    assert dense['acc'] == 0.9655
    assert dense['sp'] == 0.9764
    assert dense['se'] == 0.9145


def test_val_blocks_sorted_by_epoch_regardless_of_log_order():
    blocks = parse_val_blocks(DENSE_VAL_LINE + SPARSE_VAL_LINE)
    assert [b['epoch'] for b in blocks] == [1, 2]


def test_test_summary_reassembles_multiline_confusion_matrix():
    test = parse_test_summary(TEST_SUMMARY_LINE)
    assert test is not None
    assert test['TN'] == 31403838
    assert test['FP'] == 614506
    assert test['FN'] == 790663
    assert test['TP'] == 6512593
    # precision isn't logged directly -- recovered as TP / (TP + FP), matching
    # notebooks/train_test.ipynb cell 19's identical recovery.
    assert abs(test['prec'] - 6512593 / (6512593 + 614506)) < 1e-9
    assert test['dsc'] == 0.9026


def test_test_summary_missing_returns_none():
    assert parse_test_summary('no test has run yet\n') is None


def test_test_summary_uses_last_match_when_rerun():
    first = TEST_SUMMARY_LINE
    second = first.replace('f1_or_dsc: 0.9026', 'f1_or_dsc: 0.9100')
    test = parse_test_summary(first + second)
    assert test['dsc'] == 0.9100


def test_per_image_dice_extraction():
    text = (
        'test image 0: dice: 0.8358\n'
        'test image 1: dice: 0.8653\n'
        'test image 12: dice: 0.9497\n'
    )
    assert parse_per_image_test_dice(text) == [(0, 0.8358), (1, 0.8653), (12, 0.9497)]


def test_per_image_dice_absent_returns_empty_list():
    assert parse_per_image_test_dice(SPARSE_VAL_LINE + TEST_SUMMARY_LINE) == []


def test_epoch_summary_line_preferred_over_iter_fallback():
    text = (
        'train: epoch 1, iter:0, loss: 2.3993, lr: 0.001\n'
        'train: epoch 1, iter:20, loss: 1.0940, lr: 0.001\n'
        'train epoch: 1 done, mean loss: 1.0500\n'
    )
    assert parse_train_epoch_summary(text) == {1: 1.0500}
    assert derive_train_loss_per_epoch(text) == {1: 1.0500}


def test_epoch_loss_falls_back_to_last_iter_line_for_old_logs():
    # Old logs (pre-dating the 'train epoch: N done' summary line) have only
    # print_interval-gated iter lines. The fallback uses each epoch's LAST such
    # line -- the running mean over the most iterations seen in that epoch.
    text = (
        'train: epoch 1, iter:0, loss: 2.3993, lr: 0.001\n'
        'train: epoch 1, iter:20, loss: 1.0940, lr: 0.001\n'
        'train: epoch 2, iter:0, loss: 0.9000, lr: 0.0009\n'
    )
    assert parse_train_epoch_summary(text) == {}
    assert derive_train_loss_per_epoch(text) == {1: 1.0940, 2: 0.9000}


def test_lr_per_epoch_takes_first_iter_line_scientific_notation():
    # lr is constant within an epoch (scheduler.step() runs once, after the
    # loop), and can be logged in scientific notation for small values.
    text = (
        'train: epoch 9, iter:0, loss: 0.30, lr: 9.995065603657316e-05\n'
        'train: epoch 9, iter:20, loss: 0.29, lr: 9.995065603657316e-05\n'
    )
    assert parse_lr_per_epoch(text) == {9: 9.995065603657316e-05}


def test_find_latest_run_picks_most_recently_modified(tmp_path):
    older = tmp_path / 'UltraLight_VM_UNet_ISIC2017_run_a'
    newer = tmp_path / 'UltraLight_VM_UNet_ISIC2017_run_b'
    older.mkdir()
    newer.mkdir()
    # Make mtimes unambiguous rather than relying on directory-creation order.
    os.utime(older, (1_000_000, 1_000_000))
    os.utime(newer, (2_000_000, 2_000_000))
    assert find_latest_run(str(tmp_path)) == str(newer)


def test_find_latest_run_raises_when_empty(tmp_path):
    empty = tmp_path / 'empty'
    empty.mkdir()
    try:
        find_latest_run(str(empty))
        assert False, 'expected SystemExit for a results dir with no runs'
    except SystemExit:
        pass
