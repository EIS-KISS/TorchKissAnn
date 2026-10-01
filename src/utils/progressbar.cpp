//
// TorchKissAnn - A collection of tools to train various types of Machine learning
// algorithms on various types of EIS data
// Copyright (C) 2025 Carl Klemm <carl@uvos.xyz>
//
// This file is part of TorchKissAnn.
//
// TorchKissAnn is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// TorchKissAnn is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with TorchKissAnn.  If not, see <http://www.gnu.org/licenses/>.
//

#include "progressbar.h"

#include "globals.h"

Progress::Progress(const std::string& prefix, size_t maxProgress)
{
	if(!progress_bars)
		return;

	indicators::show_console_cursor(false);
	bar.emplace(
		indicators::option::BarWidth(50),
		indicators::option::PrefixText(prefix),
		indicators::option::ShowElapsedTime(true),
		indicators::option::ShowRemainingTime(true),
		indicators::option::MaxProgress(maxProgress)
	);
}

Progress::~Progress()
{
	if(!bar)
		return;

	// destroy the bar before the cursor comes back, it writes its final line
	bar.reset();
	indicators::show_console_cursor(true);
}

void Progress::tick()
{
	if(bar)
		bar->tick();
}

void Progress::setProgress(size_t progress)
{
	if(bar)
		bar->set_progress(progress);
}

void Progress::markAsCompleted()
{
	if(bar)
		bar->mark_as_completed();
}
