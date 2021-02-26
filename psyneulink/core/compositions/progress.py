import time
from rich import print
from rich.progress import Progress


class PNLProgress:
    """
    A singleton context wrapper around rich progress bars. It returns the currently active progress context instance
    if one has been instantiated already in another scope. It deallocates the progress bar when the outermost
    context is released. This class could be extended to return any object that implements the subset of rich's task
    based progress API.
    """
    _instance = None

    def __new__(cls, disable: bool = False) -> 'PNLProgress':
        if cls._instance is None:
            cls._instance = super(PNLProgress, cls).__new__(cls)

            # Instantiate a rich progress context\object. It is not started yet.
            cls._instance._progress = Progress(disable=disable)

            # This counter is increments on each context __enter__ and decrements
            # on each __exit__. We need this to make sure we don't call progress
            # stop until all references have been released.
            cls._ref_count = 0

        return cls._instance

    @classmethod
    def _destroy(cls) -> None:
        """
        A simple helper method that deallocates the singleton instance. This is called when we want to fully destroy
        the singleton instance and its member progress counters. This will cause the next call to PNLProgress() to
        create a completely new singleton instance.
        """
        cls._instance = None

    def __enter__(self) -> Progress:
        """
        This currently returns a singleton of the rich.Progress class.
        Returns:
            A new singleton rich progress context if none is currently active, otherwise, it returns the currently
            active context.
        """

        # If this is the top level call to with PNLProgress(), start the progress
        if self._ref_count == 0 and not self._progress.disable:
            self._progress.start()

        # Keep track of a reference count of how many times we have given a reference.
        self._ref_count = self._ref_count + 1

        return self._progress

    def __exit__(self, type, value, traceback) -> None:
        """
        Called when the context is closed.
        Args:
            type:
            value:
            traceback:
        Returns:
            Returns None so that exceptions generated within the context are propogated back up
        """

        # We are releasing this reference
        self._ref_count = self._ref_count - 1

        # If all references are released, stop progress reporting and destroy the singleton.
        if self._ref_count == 0:

            # If the progress bar is not disabled, stop it.
            if not self._progress.disable:
                self._progress.stop()

            # Destroy the singleton, very important. If we don't do this, the rich progress
            # bar will grow and grow and never be deallocated until the end of program.
            PNLProgress._destroy()

####################################################
# An Example
####################################################


def another_run(task_num):
    """A simple function to generate another task progress bar."""
    with PNLProgress() as progress:
        task = progress.add_task(f"[white]Another Task {task_num} ...", total=100)

        for i in range(100):
            progress.update(task, advance=1)
            time.sleep(0.001)

        # We can remove a task if we want to. I notice rich gets pretty slow and can even cause the terminal to hang
        # if a large number of tasks are created so you might need to do this. It seems to get bad around 100 or so
        # tasks on my machine.
        progress.remove_task(task)


def run(show_progress: bool = True):

    with PNLProgress(disable=not show_progress) as progress:

        task1 = progress.add_task("[red]Downloading...", total=100)
        task2 = progress.add_task("[green]Processing...", total=100)
        task3 = progress.add_task("[cyan]Cooking...", total=100)

        i = 0
        sub_task_num = 0

        while not progress.finished:
            progress.update(task1, advance=0.5)
            progress.update(task2, advance=0.3)
            progress.update(task3, advance=0.9)
            time.sleep(0.002)

            # Run another whole task every 30 iterations
            if i % 30 == 0:
                sub_task_num = sub_task_num + 1
                another_run(sub_task_num)

            i = i + 1

run()

print("Run Again ... Progress Disabled")
run(show_progress=False)

print("Run Again")
run()