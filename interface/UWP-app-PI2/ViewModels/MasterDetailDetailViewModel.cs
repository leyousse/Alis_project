using System;

using Caliburn.Micro;

using Demoo.Core.Models;

namespace Demoo.ViewModels
{
    public class MasterDetailDetailViewModel : Screen
    {
        public MasterDetailDetailViewModel(SampleOrder item)
        {
            Item = item;
        }

        public SampleOrder Item { get; }
    }
}
