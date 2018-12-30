---
layout: post
title: "Deepfakes - Is Seeing Still Believing?"
excerpt: "Has widespread misuse of AI arrived? Not quite yet..."
author: joshua_morton
editor: andrey_kurenkov
tags: [society,panic,DeepFakes]
redirect_from: /content/news/deepfakes/
permalink: /briefs/deepfakes/
---

The first high-profile example of state of the art AI techniques causing large
scale harm, and likely far from the last. But it may not be as bad as it
seems.

## What Happened

In late September 2017, a reddit user by the name of "deepfakes" appeared. By
February 2018, the activity of this deepfakes had been covered by Forbes, the
BBC, the NYT, and a multitude of tech news sources.

The cause of this coverage? Deepfakes, as the name implies, used
[deep](http://theai.wiki/Deep%20Learning) [neural
networks](http://theai.wiki/Neural%20Network) to generate fake (but at a glance,
convincing) images and videos of porn in which the faces of the pornstars was
swapped with the faces of various celebrities.  

The images and videos look highly realistic. Other people, like one named
"derpfakes" used the same methods for more amusing results things like pasting
Angela Merkel's face onto Donald Trump, or sticking Nicholas Cage into every
film role ever:

<figure>
	<img src="{{site_url}}/content/news/images/deepfakes/cage.jpg" alt="Derpfakes Example">
	<figcaption>
A deepfake, created by a user named "derpfakes", which superimposes Nicholas
Cage's face onto Star Trek: The Next Generation's Captain Picard
	</figcaption>
</figure>


It's important to discuss the research vs. the application in this case. The
original deepfakes appears to be based on a simplified version of [this paper
from NVIDIA](https://arxiv.org/abs/1703.00848). It however doesn't use
[Generative Aversarial Networks
(GANs)](http://theai.wiki/Generative%20Adversarial%20Network%20%28GAN%29),
instead just taking advantage of a dual
[autoencoder](http://theai.wiki/Autoencoder) approach.[^autoencoder] More recent
versions of the Deepfakes code do take advantage of GANs[^GAN]. Of course, the
methods outlined in these papers were not developed with these uses in mind, but
the availibility of papers and code does enable such misuses.

## The Reactions

Vice's motherboard broke the story of deepfakes with a headline reading ["We're
All
Fucked"](https://motherboard.vice.com/en_us/article/gydydm/gal-gadot-fake-ai-porn),
in December. It focused on a worrying video where the face of Gal Gadot, star of
DC's Wonder Woman, was superimposed onto the body of a pornstar.

It took another month for the coverage to really explode, though. Motherboard
again broke the story that deepfakes had been made into an accessible app by a
tool called "FakeApp"; this time the headline read ["We are Truly Fucked,
Everyone Is Making AI-Generated Fake Porn
Now"](https://motherboard.vice.com/en_us/article/bjye8a/reddit-fake-porn-app-daisy-ridley).
While previously creation of deepfakes content was limited to programmers and experts,
FakeApp heavily automates the process and so allows laypeople to
create the faceswapped videos with relative ease.  

In that second article, Vice says:

> It isn’t difficult to imagine an amateur programmer running their own
> algorithm to create a sex tape of someone they want to harass.

[Lawfare](https://lawfareblog.com/deep-fakes-looming-crisis-national-security-democracy-and-privacy)
goes further, saying

> The spread of deep fakes will threaten to erode the trust necessary for
> democracy to function effectively

And other sources added their own opinions, with 
[The Verge](https://www.theverge.com/2018/1/24/16929148/fake-celebrity-porn-ai-deepfake-face-swapping-artificial-intelligence-reddit),
[Forbes](https://www.forbes.com/sites/ianmorris/2018/02/05/fakeapp-allows-anyone-to-make-deepfake-porn-of-anyone/#1cbb82b7391c),
[The Outline](https://theoutline.com/post/3179/deepfake-videos-are-freaking-experts-out?zd=2&zi=7uvt66te),
and the [BBC](http://www.bbc.com/news/technology-42912529)
describing ways that deepfakes could lead to revenge porn, blackmail, and
worstening fake news.

> What about blackmail? Find a girl with suitably conservative and perhaps
> deeply religious parents and fake her into a porn video.

> The fake news crisis, as we know it today, may only just be the beginning.

> the “nightmare situation of somebody creating a video of Trump saying ‘I’ve
> launched nuclear weapons against North Korea,’

In the wake of this coverage, on February 7, Reddit [updated its
rules](https://www.reddit.com/r/announcements/comments/7vxzrb/update_on_sitewide_rules_regarding_involuntary/)
about involuntary pornography, outlawing ["any person in a state of nudity [...]
posted without their permission, including depictions that have been
faked"](https://www.reddithelp.com/en/categories/rules-reporting/account-and-community-restrictions/do-not-post-involuntary-pornography).
This caused discussion across
[Reddit](https://www.reddit.com/r/SubredditDrama/comments/7vy9cw/rdeepfakes_the_aigenerated_fake_celebrity_porn/)
and [Hacker News](https://news.ycombinator.com/item?id=16327489) about questions
of everything from ethics to speech to intellectual property law. Those changes
seem to have served their purpose: deepfakes appears to have left the spotlight,
at least for now.

## Our Perspective

So how bad is this - are we all fucked, really?

It's not good, and for a change the media coverage about this story is largely
solid. Still, there are a few places where non-techincal journalists get carried
away and overestimate what the tools are capable of now. 

While the ability of generative tools to create realistic fakes has vastly
improved over the past 5 or so years, they aren't yet all the way there,
and currently, are either relatively easy to spot as fake, or not able to
produce the kinds of images deepfakes demands.

<center>
<blockquote class="twitter-tweet" data-lang="en"><p lang="en" dir="ltr">4 years
of GAN progress (source: <a
href="https://t.co/hlxW3NnTJP">https://t.co/hlxW3NnTJP</a> ) <a
href="https://t.co/kmK5zikayV">pic.twitter.com/kmK5zikayV</a></p>&mdash; Ian
Goodfellow (@goodfellow_ian) <a
href="https://twitter.com/goodfellow_ian/status/969776035649675265?ref_src=twsrc%5Etfw">March
3, 2018</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js"
charset="utf-8"></script>
</center>

What's more, state of the art methods for video generation rely on a paired set
of models, one that generates fakes, and one that recognizes fakes (the
aforementioned "GAN"). If GANs continue to be the state of the art in fake
generation, we know that it is at least possible for other ML systems to
differentiate between fake and real images.[^GAN2] This will continue to be true
even if it becomes impossible for expert humans to distinguish the fakes from
real video; and at least for now, deepfakes are straightforward to spot.

Many of the articles also suggest faked revenge porn or blackmail as possible
outcomes. While that shouldn't be wholly discounted, that specific application
is not as easy as claimed. Deepfakes, like most machine learning models,
requires a lot of data. Making a fake requires a significant number of good
quality images of a person. For celebrities and politicians, this is easy to
come by. But for a use case like personal blackmail, it's less likely that
there will be a lot of good images or videos availible. [^oneshot]

A few examples: it took thousands of professional-quality headshots for a
[NYT reporter to get decent results with
FaceApp](https://www.nytimes.com/2018/03/04/technology/fake-videos-deepfakes.html)
, and nearly one thousand for [Sven
Charlier](http://svencharleer.com/blog/2018/02/02/family-fun-with-deepfakes-or-how-i-got-my-wife-onto-the-tonight-show/),
an HCI researcher, to get reasonable results at a relatively low resolution.
That implies that at least for the forseeable future, most deepfaked blackmail will
be low resolution and grainy. Perhaps more importantly, because of these
limitations, [right now it is possible to detect many celebrity deepfakes 
since they use existing video and celebrity
photos](https://www.wired.com/story/gfycat-artificial-intelligence-deepfakes/).

Another important thing to note about how deepfakes may affect fake news is that
there are already many ways to create convincing fake videos. One exceptionally
silly example from lawfare was that deepfakes could be used to "falsely depict a
white police officer shooting an unarmed black man while shouting racial
epithets". Creating such a video is already possible, no machine learning
necessary.

These corrections aside though, for the most part the coverage has been correct
and this is indeed a worrying turn in the [Deep
Learning](http://theai.wiki/Deep%20Learning)-initiated AI boom we are still in
the middle of.

## TLDR

Pornography with the use of AI techniques is certainly an ethical concern and
bringing it to light is important. However, media outlets probably overreach
when they suggest that you might be next.


*Josh is currently employed by Google, but these opinions are his own.*

[^autoencoder]: In short, this approach works by taking two autoencoders and forcing them to share the same "latent space". In other words, you train an autoencoder to compress and then decompress your face, and train an autoencoder to compress and then decompress a celebrity face. By being clever and sharing the encoder between both models, you make sure that the compressed representations are similar, so then you can attach the decoder from the celebrity face to the shared encoder, and then pass in your face and get out a celebrity face.
[^GAN]: [CycleGAN](https://github.com/junyanz/CycleGAN), and [pix2pix](https://github.com/phillipi/pix2pix) are two examples of GAN models modified for [faceswapping](https://github.com/shaoanlu/faceswap-GAN). In general, GANs work by training a pair of models, a discriminator and a generator. The Generator tries to trick the discriminator, and the discriminator in turn tries to detect the fakes produced by the generator.
[^GAN2]: Since the GAN works by pairing a discriminator and a generator, the existence of a good generator is only possible due to an equally good discriminator. The problem with this is in practice is that there is no reason for a deepfake creator to publish the discriminator used as part of training.
[^oneshot]: Machine learning with a small amount of training data is an active area of research known as "one-shot learning". Current one-shot learning techniques can't generate the types of images that are required for convincing deepfakes.
