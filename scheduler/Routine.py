from pathlib import Path
from configuration.config import FINISH_STAGE
from trainers.BaseTrainer import AbstractBaseTrainer
import logging

class Routine:
    def __init__(self, routine_list: list[str], trainer_list: list[AbstractBaseTrainer], args, export_root: Path) -> None:
        self._routine_list = routine_list
        self._trainer_list = trainer_list
        self._export_root = export_root

        self._set_up(args)

    def _set_up(self, args):
        if FINISH_STAGE not in self._routine_list:
            self._routine_list.append(FINISH_STAGE)
        else:
            assert(len(self._routine_list) > 1)

        self._current_routine = self._routine_list[0]

        self._next_routine = {cur: nxt for (cur, nxt) in zip(self._routine_list[:-1], self._routine_list[1:])}
        self._trainer_dict = {routine: trainer for (routine, trainer) in zip(self._routine_list[:-1], self._trainer_list)}

        self._set_current_routine(args)

    def _set_current_routine(self, args):
        routine = args.training_routine
        if routine is not None and routine in self._routine_list and routine is not FINISH_STAGE:
            self._current_routine = routine
        
        logging.info(f"first routine: {self._current_routine}")

    def _nxt_routine(self):
        assert(self._current_routine != FINISH_STAGE)

        self._current_routine = self._next_routine[self._current_routine]

        # return self._current_routine

    def run_routine(self):
        while self._current_routine != FINISH_STAGE:
            logging.info(f"Start routine {self._current_routine}")

            logging.info(f"Start training")

            self._trainer_dict[self._current_routine].train()

            logging.info(f"Finished training, Start testing")

            result = self._trainer_dict[self._current_routine].test(self._export_root)

            logging.info(f"Finished testing. Result: {result}")

            self._nxt_routine()